import json
import os
import re
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    import httpx
except ImportError:
    httpx = None


# MODEL_PATH = os.getenv("EUI_MODEL_PATH", "models_sklearn_wwr_split/eui_ann_model.joblib")
# SCALER_PATH = os.getenv("EUI_SCALER_PATH", "models_torch/eui_ann_torch_scalers.joblib")
MODEL_PATH = os.getenv("EUI_MODEL_PATH", "models_torch_range2_physics_1/eui_ann_torch_model.pt")
SCALER_PATH = os.getenv("EUI_SCALER_PATH", "models_torch_range2_physics_1/eui_ann_torch_scalers.joblib")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

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
    "wall_r": float(os.getenv("EUI_BASELINE_WALL_R", 31.0)),
    "roof_r": float(os.getenv("EUI_BASELINE_ROOF_R", 49.0)),
    "window_u": float(os.getenv("EUI_BASELINE_WINDOW_U", 0.35)),
    "wwr": float(os.getenv("EUI_BASELINE_WWR", 0.30)),
    "shgc": float(os.getenv("EUI_BASELINE_SHGC", 0.40)),
    "lighting_density": float(os.getenv("EUI_BASELINE_LIGHTING_DENSITY", 4.0)),
    "infiltration": float(os.getenv("EUI_BASELINE_INFILTRATION", 0.00045)),
    "orientation_deg": float(os.getenv("EUI_BASELINE_ORIENTATION_DEG", 180.0)),
}


try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class TorchEUIRegressor(nn.Module if nn is not None else object):
    def __init__(self, input_dim, hidden_layers, activation):
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "logistic": nn.Sigmoid,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[activation]())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DesignInput(BaseModel):
    wall_r: float = Field(..., description="Wall assembly R-value, IP units")
    roof_r: float = Field(..., description="Roof assembly R-value, IP units")
    window_u: float = Field(..., description="Window assembly U-factor, IP units")
    wwr: float = Field(..., description="Window-to-wall ratio")
    shgc: float = Field(..., description="Solar heat gain coefficient")
    lighting_density: float = Field(
        BASELINE_DESIGN["lighting_density"],
        description="Lighting power density, W/m2",
    )
    infiltration: float = Field(..., description="Infiltration rate, m3/s per m2 facade")
    orientation_deg: float = Field(
        BASELINE_DESIGN["orientation_deg"],
        description="Building orientation in degrees",
    )


class SuggestionRequest(BaseModel):
    design: DesignInput
    target_eui: Optional[float] = Field(None, description="Optional target EUI")
    n_candidates: int = Field(2000, ge=100, le=20000)
    top_k: int = Field(5, ge=1, le=20)
    fixed_parameters: List[str] = Field(
        default_factory=list,
        description="Design parameter keys that should remain unchanged during candidate search.",
    )


class AgentRequest(BaseModel):
    description: str = Field(..., min_length=3, description="Natural-language apartment design description")


app = FastAPI(
    title="Apartment EUI ANN API",
    description="Predict annual EUI from apartment design parameters.",
    version="0.1.0",
)


def parse_cors_origins() -> List[str]:
    raw = os.getenv("EUI_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


artifact = None


def get_anthropic_api_key() -> str:
    raw_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key = "".join(raw_key.strip().strip('"').strip("'").split())
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set. Set it in the API server environment variables.",
        )
    if not api_key.startswith("sk-ant-"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY appears to be invalid. It should start with sk-ant-.",
        )
    return api_key


def load_artifact():
    global artifact
    if artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found: {MODEL_PATH}. Train the model first.",
            )
        if MODEL_PATH.endswith(".joblib"):
            artifact = joblib.load(MODEL_PATH)
            artifact["model_type"] = "sklearn"
        elif MODEL_PATH.endswith(".pt"):
            if torch is None:
                raise HTTPException(
                    status_code=503,
                    detail="PyTorch is not installed in this Python environment.",
                )
            if not os.path.exists(SCALER_PATH):
                raise HTTPException(
                    status_code=503,
                    detail=f"Scaler file not found: {SCALER_PATH}.",
                )
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            scalers = joblib.load(SCALER_PATH)
            model = TorchEUIRegressor(
                input_dim=checkpoint["input_dim"],
                hidden_layers=checkpoint["hidden_layers"],
                activation=checkpoint["activation"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            artifact = {
                "model_type": "torch",
                "model": model,
                "x_scaler": scalers["x_scaler"],
                "y_scaler": scalers["y_scaler"],
                "feature_columns": checkpoint["feature_columns"],
                "target_column": checkpoint["target_column"],
                "feature_bounds": checkpoint["metrics"]["feature_bounds"],
                "metrics": checkpoint["metrics"],
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Unsupported model file. Use .joblib for sklearn or .pt for PyTorch.",
            )
    return artifact


@app.get("/")
def root():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(
            f.read(),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )


@app.get("/api")
def api_index():
    return {
        "message": "Apartment EUI ANN API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "suggest": "/suggest",
        "agent": "/agent",
    }


def design_to_dataframe(design: DesignInput, feature_columns: List[str]) -> pd.DataFrame:
    orientation_rad = np.deg2rad(design.orientation_deg)
    values_by_feature = {
        "Wall Assembly R-Value (f-ft2-F/Btu)": design.wall_r,
        "Roof Assembly R-Value (f-ft2-F/Btu)": design.roof_r,
        "Window Assembly U-Factor (Btu/f-ft2-F)": design.window_u,
        "Window-to-Wall Ratio": design.wwr,
        "SHGC": design.shgc,
        "Lighting Density (W/m2)": design.lighting_density,
        "Infiltration Rate (m3/s per m2 of facade)": design.infiltration,
        "Infiltration Rate (m3/s per m2 of façade)": design.infiltration,
        "Orientation sin": float(np.sin(orientation_rad)),
        "Orientation cos": float(np.cos(orientation_rad)),
    }
    return pd.DataFrame([{col: values_by_feature[col] for col in feature_columns}])


def predict_eui_from_dataframe(df: pd.DataFrame, model_artifact: Dict) -> np.ndarray:
    x = model_artifact["x_scaler"].transform(df[model_artifact["feature_columns"]].to_numpy(dtype=float))
    if model_artifact["model_type"] == "sklearn":
        pred_scaled = model_artifact["model"].predict(x).reshape(-1, 1)
    else:
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            pred_scaled = model_artifact["model"](tensor).numpy().reshape(-1, 1)
    return model_artifact["y_scaler"].inverse_transform(pred_scaled).ravel()


def validate_bounds(df: pd.DataFrame, model_artifact: Dict):
    warnings = []
    for col, bounds in model_artifact["feature_bounds"].items():
        value = float(df.iloc[0][col])
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{col}={value:.6g} is outside training range "
                f"[{bounds['min']:.6g}, {bounds['max']:.6g}]"
            )
    return warnings


@app.get("/health")
def health():
    model_artifact = load_artifact()
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_type": model_artifact["model_type"],
        "n_samples": model_artifact["metrics"]["n_samples"],
        "test_metrics": model_artifact["metrics"]["test"],
    }


@app.get("/bounds")
def bounds():
    model_artifact = load_artifact()
    return model_artifact["feature_bounds"]


@app.post("/predict")
def predict(design: DesignInput):
    model_artifact = load_artifact()
    df = design_to_dataframe(design, model_artifact["feature_columns"])
    pred = float(predict_eui_from_dataframe(df, model_artifact)[0])
    baseline_design = DesignInput(**BASELINE_DESIGN)
    baseline_df = design_to_dataframe(baseline_design, model_artifact["feature_columns"])
    baseline_pred = float(predict_eui_from_dataframe(baseline_df, model_artifact)[0])
    p_eui_reduction = (baseline_pred - pred) / baseline_pred if baseline_pred else None
    return {
        "predicted_eui_kbtu_sf_yr": pred,
        "baseline_eui_kbtu_sf_yr": baseline_pred,
        "p_eui_reduction": p_eui_reduction,
        "baseline_design": BASELINE_DESIGN,
        "warnings": validate_bounds(df, model_artifact),
        "model_test_metrics": model_artifact["metrics"]["test"],
    }


def call_claude(prompt: str, max_tokens: int = 1200) -> str:
    if httpx is None:
        raise HTTPException(
            status_code=503,
            detail="httpx is not installed in this Python environment. Install it to use the Claude agent.",
        )

    api_key = get_anthropic_api_key()

    headers = {
        "x-api-key": api_key,
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
            data = response.json()
    except httpx.HTTPStatusError as exc:
        response_text = exc.response.text
        if exc.response.status_code in {401, 403}:
            response_text = "Authentication failed. Check the ANTHROPIC_API_KEY environment variable."
        raise HTTPException(
            status_code=502,
            detail=(
                f"Anthropic API request failed with status {exc.response.status_code}: "
                f"{response_text}"
            ),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Anthropic API request failed before a response was returned. Check the API key formatting and server logs.",
        ) from exc

    return data["content"][0]["text"]


def extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise HTTPException(
            status_code=502,
            detail=f"Claude did not return JSON for the design parameters. Raw response: {text}",
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Claude returned invalid JSON for the design parameters. Raw response: {text}",
        ) from exc


def design_from_text(description: str) -> DesignInput:
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
Do not include comments or markdown.

User description:
{description}
"""
    return DesignInput(**extract_json(call_claude(prompt, max_tokens=350)))


def explain_agent_result(description: str, design: DesignInput, prediction: Dict) -> str:
    predicted = prediction["predicted_eui_kbtu_sf_yr"]
    baseline = prediction["baseline_eui_kbtu_sf_yr"]
    reduction = prediction.get("p_eui_reduction")
    warnings = prediction.get("warnings", [])
    reduction_text = "not available" if reduction is None else f"{reduction * 100:.1f}%"
    warning_text = "Inputs are inside the training range." if not warnings else " | ".join(warnings)
    return f"""## Predicted EUI
- Predicted EUI: {predicted:.2f} kBtu/sf/yr.
- pEUI reduction vs. baseline: {reduction_text}; baseline EUI is {baseline:.2f} kBtu/sf/yr.
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


@app.post("/agent")
def agent(request: AgentRequest):
    design = design_from_text(request.description)
    prediction = predict(design)
    response_text = explain_agent_result(request.description, design, prediction)
    return {
        "description": request.description,
        "design": design.dict(),
        "prediction": prediction,
        "response": response_text,
        "agent_model": ANTHROPIC_MODEL,
    }


@app.post("/suggest")
def suggest(request: SuggestionRequest):
    model_artifact = load_artifact()
    feature_columns = model_artifact["feature_columns"]
    current_df = design_to_dataframe(request.design, feature_columns)
    current_eui = float(predict_eui_from_dataframe(current_df, model_artifact)[0])
    allowed_fixed_parameters = set(PARAMETER_BOUNDS)
    fixed_parameters = set(request.fixed_parameters)
    unknown_fixed_parameters = sorted(fixed_parameters - allowed_fixed_parameters)
    if unknown_fixed_parameters:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown fixed parameters: {unknown_fixed_parameters}",
        )

    rng = np.random.default_rng(42)
    candidates = []
    for _ in range(request.n_candidates):
        candidate = {}
        for field_name, feature_name in [
            ("wall_r", "Wall Assembly R-Value (f-ft2-F/Btu)"),
            ("roof_r", "Roof Assembly R-Value (f-ft2-F/Btu)"),
            ("window_u", "Window Assembly U-Factor (Btu/f-ft2-F)"),
            ("wwr", "Window-to-Wall Ratio"),
            ("shgc", "SHGC"),
            ("lighting_density", "Lighting Density (W/m2)"),
            ("infiltration", next(col for col in feature_columns if col.startswith("Infiltration Rate"))),
        ]:
            if feature_name not in feature_columns:
                continue
            current_value = float(getattr(request.design, field_name))
            if field_name in fixed_parameters:
                candidate[field_name] = current_value
            else:
                bounds = model_artifact["feature_bounds"][feature_name]
                span = bounds["max"] - bounds["min"]
                low = max(bounds["min"], current_value - 0.20 * span)
                high = min(bounds["max"], current_value + 0.20 * span)
                candidate[field_name] = rng.uniform(low, high)
        if "orientation_deg" in fixed_parameters:
            candidate["orientation_deg"] = request.design.orientation_deg
        else:
            orientation_low, orientation_high = PARAMETER_BOUNDS["orientation_deg"]
            orientation_span = orientation_high - orientation_low
            low = max(orientation_low, request.design.orientation_deg - 0.20 * orientation_span)
            high = min(orientation_high, request.design.orientation_deg + 0.20 * orientation_span)
            candidate["orientation_deg"] = rng.uniform(low, high)
        candidates.append(design_to_dataframe(DesignInput(**candidate), feature_columns).iloc[0].to_dict())

    candidate_df = pd.DataFrame(candidates)
    candidate_df["predicted_eui_kbtu_sf_yr"] = predict_eui_from_dataframe(candidate_df, model_artifact)
    candidate_df["improvement_kbtu_sf_yr"] = current_eui - candidate_df["predicted_eui_kbtu_sf_yr"]

    if request.target_eui is not None:
        candidate_df = candidate_df[candidate_df["predicted_eui_kbtu_sf_yr"] <= request.target_eui]

    candidate_df = candidate_df.sort_values(
        ["predicted_eui_kbtu_sf_yr", "improvement_kbtu_sf_yr"],
        ascending=[True, False],
    ).head(request.top_k)

    suggestions = []
    for _, row in candidate_df.iterrows():
        candidate_eui = float(row["predicted_eui_kbtu_sf_yr"])
        suggestions.append({
            "wall_r": float(row["Wall Assembly R-Value (f-ft2-F/Btu)"]),
            "roof_r": float(row["Roof Assembly R-Value (f-ft2-F/Btu)"]),
            "window_u": float(row["Window Assembly U-Factor (Btu/f-ft2-F)"]),
            "wwr": float(row["Window-to-Wall Ratio"]),
            "shgc": float(row["SHGC"]),
            "lighting_density": float(row["Lighting Density (W/m2)"]) if "Lighting Density (W/m2)" in row else None,
            "infiltration": float(row[next(col for col in feature_columns if col.startswith("Infiltration Rate"))]),
            "orientation_deg": float((np.rad2deg(np.arctan2(row.get("Orientation sin", 0.0), row.get("Orientation cos", 1.0))) + 360.0) % 360.0),
            "predicted_eui_kbtu_sf_yr": candidate_eui,
            "improvement_kbtu_sf_yr": float(row["improvement_kbtu_sf_yr"]),
            "p_eui_reduction": (current_eui - candidate_eui) / current_eui if current_eui else None,
        })

    return {
        "current_predicted_eui_kbtu_sf_yr": current_eui,
        "suggestions": suggestions,
        "fixed_parameters": sorted(fixed_parameters),
        "warnings": validate_bounds(current_df, model_artifact),
    }
