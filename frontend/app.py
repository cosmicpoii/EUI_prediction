import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MODEL_PATH = Path(__file__).with_name("eui_model_numpy.npz")

PARAMETER_BOUNDS = {
    "wall_r": [9.0117, 56.9917],
    "roof_r": [17.0299, 73.9716],
    "window_u": [0.1254, 0.5696],
    "wwr": [0.10, 0.4997],
    "shgc": [0.2004, 0.6998],
    "lighting_density": [0.4633, 7.6596],
    "infiltration": [0.00025, 0.00075],
    "orientation_deg": [0.0, 360.0],
}

BASELINE_DESIGN = {
    "wall_r": 31.0,
    "roof_r": 49.0,
    "window_u": 0.35,
    "wwr": 0.30,
    "shgc": 0.40,
    "lighting_density": 4.0,
    "infiltration": 0.00045,
    "orientation_deg": 180.0,
}


class DesignInput(BaseModel):
    wall_r: float
    roof_r: float
    window_u: float
    wwr: float
    shgc: float
    lighting_density: float = BASELINE_DESIGN["lighting_density"]
    infiltration: float
    orientation_deg: float = BASELINE_DESIGN["orientation_deg"]


class SuggestionRequest(BaseModel):
    design: DesignInput
    target_eui: Optional[float] = None
    n_candidates: int = Field(2000, ge=100, le=20000)
    top_k: int = Field(5, ge=1, le=20)
    fixed_parameters: List[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    description: str = Field(..., min_length=3)


app = FastAPI(
    title="Apartment EUI ANN API",
    description="Lightweight NumPy inference for the final lambda=0.1 ANN.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@lru_cache(maxsize=1)
def load_artifact():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    archive = np.load(MODEL_PATH, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"].item()))
    return {
        "x_mean": archive["x_mean"],
        "x_scale": archive["x_scale"],
        "y_mean": archive["y_mean"],
        "y_scale": archive["y_scale"],
        "weights": [archive[f"weight_{i}"] for i in range(metadata["n_layers"])],
        "biases": [archive[f"bias_{i}"] for i in range(metadata["n_layers"])],
        **metadata,
    }


def gelu_exact(values):
    flat = values.reshape(-1)
    erf_values = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in flat),
        dtype=np.float32,
        count=flat.size,
    ).reshape(values.shape)
    return (0.5 * values * (1.0 + erf_values)).astype(np.float32)


def predict_features(features):
    artifact = load_artifact()
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = ((values - artifact["x_mean"]) / artifact["x_scale"]).astype(np.float32)
    for index, (weight, bias) in enumerate(zip(artifact["weights"], artifact["biases"])):
        values = values @ weight.T + bias
        if index < len(artifact["weights"]) - 1:
            values = gelu_exact(values)
    return values[:, 0].astype(np.float64) * artifact["y_scale"][0] + artifact["y_mean"][0]


def designs_to_features(designs):
    rows = []
    for design in designs:
        orientation = math.radians(float(design.orientation_deg))
        rows.append([
            float(design.wall_r),
            float(design.roof_r),
            float(design.window_u),
            float(design.wwr),
            float(design.shgc),
            float(design.lighting_density),
            float(design.infiltration),
            math.sin(orientation),
            math.cos(orientation),
        ])
    return np.asarray(rows, dtype=np.float64)


def feature_warnings(design):
    artifact = load_artifact()
    values = designs_to_features([design])[0]
    warnings = []
    for index, name in enumerate(artifact["feature_columns"]):
        bounds = artifact["metrics"]["feature_bounds"][name]
        value = float(values[index])
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{name}={value:.6g} is outside training range "
                f"[{bounds['min']:.6g}, {bounds['max']:.6g}]"
            )
    return warnings


def predict_design(design):
    predicted = float(predict_features(designs_to_features([design]))[0])
    baseline = DesignInput(**BASELINE_DESIGN)
    baseline_predicted = float(predict_features(designs_to_features([baseline]))[0])
    reduction = (baseline_predicted - predicted) / baseline_predicted if baseline_predicted else None
    return {
        "predicted_eui_kbtu_sf_yr": predicted,
        "baseline_eui_kbtu_sf_yr": baseline_predicted,
        "p_eui_reduction": reduction,
        "baseline_design": BASELINE_DESIGN,
        "warnings": feature_warnings(design),
        "model_test_metrics": load_artifact()["metrics"]["test"],
    }


def anthropic_api_key():
    raw_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key = "".join(raw_key.strip().strip('"').strip("'").split())
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set in Vercel.")
    if not api_key.startswith("sk-ant-"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY appears to be invalid.")
    return api_key


def call_claude(prompt, max_tokens=350):
    headers = {
        "x-api-key": anthropic_api_key(),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(ANTHROPIC_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["content"][0]["text"]
    except httpx.HTTPStatusError as exc:
        message = exc.response.text
        if exc.response.status_code in {401, 403}:
            message = "Authentication failed. Check ANTHROPIC_API_KEY in Vercel."
        raise HTTPException(
            status_code=502,
            detail=f"Anthropic API request failed with status {exc.response.status_code}: {message}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Anthropic API request failed.") from exc


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise HTTPException(status_code=502, detail="Claude did not return JSON parameters.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Claude returned invalid JSON parameters.") from exc


def design_from_text(description):
    prompt = f"""
You are preparing inputs for an apartment EUI surrogate model.

The model only supports this prototype domain:
- Seattle 3-story multifamily apartment prototype
- 18 conditioned living units, 6 per floor
- Gable-roof geometry: ridge length 36.54 m, span 19.84 m, eave height 7.92 m, ridge height 12.05 m
- Same schedules and HVAC assumptions as the training data
- Only these design parameters vary

Return ONLY valid JSON with these keys:
- wall_r
- roof_r
- window_u
- wwr
- shgc
- lighting_density
- infiltration
- orientation_deg

Use these training bounds:
{json.dumps(PARAMETER_BOUNDS, indent=2)}

Interpret common language like "large windows", "well insulated", "leaky", or "low air leakage"
as values inside these bounds. Interpret compass orientation as degrees clockwise from north.
If the user description is vague, choose conservative midpoint values.
If the user explicitly provides a numerical value, copy that numerical value exactly into the JSON.
Do not clamp, replace, correct, or move explicit user-provided numerical values into the training range.
The prediction backend will validate ranges and return warnings for out-of-range values.
For qualitative or unspecified design requests only, infer values inside the training bounds.
Do not include comments or markdown.

User description:
{description}
"""
    return DesignInput(**extract_json(call_claude(prompt)))


def agent_explanation(design, prediction):
    reduction = prediction.get("p_eui_reduction")
    reduction_text = "not available" if reduction is None else f"{reduction * 100:.1f}%"
    warnings = prediction.get("warnings", [])
    warning_text = "Inputs are inside the training range." if not warnings else " | ".join(warnings)
    return f"""## Predicted EUI
- Predicted EUI: {prediction['predicted_eui_kbtu_sf_yr']:.2f} kBtu/sf/yr.
- pEUI reduction vs. baseline: {reduction_text}; baseline EUI is {prediction['baseline_eui_kbtu_sf_yr']:.2f} kBtu/sf/yr.
- This is a surrogate model for the Seattle 3-story multifamily prototype, not a general building simulator.

## Assumed Design Parameters
- Wall R: {design.wall_r:.2f}
- Roof R: {design.roof_r:.2f}
- Window U-factor: {design.window_u:.3f}
- WWR: {design.wwr:.3f}
- SHGC: {design.shgc:.3f}
- Lighting density: {design.lighting_density:.2f} W/m2
- Infiltration: {design.infiltration:.6f} m3/s per m2 facade
- Orientation: {design.orientation_deg:.0f} deg

## Warnings
- {warning_text}

## Suggested Next Tests
- Try lowering WWR or SHGC if the design has solar gains from large windows.
- Try increasing roof/wall R-values if envelope performance is the main design lever.
- Try reducing infiltration if air leakage is high.
"""


@app.get("/api")
def api_index():
    return {"status": "ok", "runtime": "vercel-numpy", "model": "lambda=0.1"}


@app.get("/health")
@app.get("/api/health")
def health():
    artifact = load_artifact()
    return {
        "status": "ok",
        "runtime": "vercel-numpy",
        "model_type": "numpy-ann",
        "model_path": str(MODEL_PATH.name),
        "n_samples": artifact["metrics"]["n_samples"],
        "test_metrics": artifact["metrics"]["test"],
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(design: DesignInput):
    return predict_design(design)


@app.post("/agent")
@app.post("/api/agent")
def agent(request: AgentRequest):
    design = design_from_text(request.description)
    prediction = predict_design(design)
    return {
        "description": request.description,
        "design": model_dump(design),
        "prediction": prediction,
        "response": agent_explanation(design, prediction),
        "agent_model": ANTHROPIC_MODEL,
    }


@app.post("/suggest")
@app.post("/api/suggest")
def suggest(request: SuggestionRequest):
    fixed = set(request.fixed_parameters)
    unknown = sorted(fixed - set(PARAMETER_BOUNDS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown fixed parameters: {unknown}")

    current_eui = float(predict_features(designs_to_features([request.design]))[0])
    rng = np.random.default_rng(42)
    candidate_values = {name: np.empty(request.n_candidates) for name in PARAMETER_BOUNDS}
    candidate_ranges = {}
    for name, bounds in PARAMETER_BOUNDS.items():
        current = float(getattr(request.design, name))
        if name in fixed:
            candidate_values[name].fill(current)
            continue
        span = bounds[1] - bounds[0]
        low = max(bounds[0], current - 0.20 * span)
        high = min(bounds[1], current + 0.20 * span)
        candidate_ranges[name] = (low, high)

    # Preserve the original backend's candidate-first random sampling order.
    for index in range(request.n_candidates):
        for name in PARAMETER_BOUNDS:
            if name not in fixed:
                candidate_values[name][index] = rng.uniform(*candidate_ranges[name])

    candidates = [
        DesignInput(**{name: float(values[index]) for name, values in candidate_values.items()})
        for index in range(request.n_candidates)
    ]
    predicted = predict_features(designs_to_features(candidates))
    eligible = np.arange(request.n_candidates)
    if request.target_eui is not None:
        eligible = eligible[predicted <= request.target_eui]
    ranked = eligible[np.argsort(predicted[eligible])[: request.top_k]]

    suggestions = []
    for index in ranked:
        design = candidates[int(index)]
        candidate_eui = float(predicted[int(index)])
        item = model_dump(design)
        item.update({
            "predicted_eui_kbtu_sf_yr": candidate_eui,
            "improvement_kbtu_sf_yr": current_eui - candidate_eui,
            "p_eui_reduction": (current_eui - candidate_eui) / current_eui if current_eui else None,
        })
        suggestions.append(item)

    return {
        "current_predicted_eui_kbtu_sf_yr": current_eui,
        "suggestions": suggestions,
        "fixed_parameters": sorted(fixed),
        "warnings": feature_warnings(request.design),
    }
