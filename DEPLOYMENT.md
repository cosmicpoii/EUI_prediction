# Deployment Guide

## Recommended: Full-stack Vercel deployment

The production site runs the static frontend and API in one Vercel project:

```text
Browser -> Vercel CDN frontend -> same-origin FastAPI function -> NumPy ANN
                                                   -> Anthropic API (agent only)
```

This path does not call Render, so a sleeping Render service cannot delay page loading or
manual ANN prediction. The serverless API uses a lightweight NumPy export of the final
lambda=0.1 PyTorch model (`32, 32, 16`, GELU). Its predictions were checked against the
original PyTorch artifact before deployment.

### Vercel project settings

- Git repository: `cosmicpoii/EUI_prediction`
- Root Directory: `frontend`
- Framework: FastAPI (configured by `frontend/vercel.json`)
- Production branch: `main`
- Required production environment variable: `ANTHROPIC_API_KEY`

`ANTHROPIC_API_KEY` must be stored in Vercel Project Settings -> Environment Variables.
Never put it in HTML, JavaScript, Git, or a `vercel.json` file. Manual prediction and
optimization do not need this key; only the natural-language agent does.

### Model export

If the selected PyTorch model changes, regenerate the Vercel artifact from the repository
root with an environment that has PyTorch and joblib installed:

```bash
python export_vercel_model.py \
  --model ../models_torch_range2_physics_01/eui_ann_torch_model.pt \
  --scalers ../models_torch_range2_physics_01/eui_ann_torch_scalers.joblib \
  --output frontend/eui_model_numpy.npz
```

Commit the regenerated `frontend/eui_model_numpy.npz` together with any API changes.

### Deploy

Pushing `main` triggers the connected Vercel production deployment. A manual deployment
from this repository is also possible:

```bash
npx vercel --prod --yes
```

Verify these routes after deployment:

```text
https://your-site.vercel.app/
https://your-site.vercel.app/api/health
https://your-site.vercel.app/api/docs
```

The frontend detects a `*.vercel.app` host and uses `/api` automatically. `?api=...`
remains available as an explicit API override for local testing or recovery.

## Render fallback

The existing Render service remains a fallback and continues to use the full PyTorch
backend. It is not in the normal Vercel request path. A free Render instance may sleep
after inactivity, so its first request can take substantially longer than subsequent
requests.

Render environment variables:

```text
EUI_MODEL_PATH=models_torch_range2_physics_01/eui_ann_torch_model.pt
EUI_SCALER_PATH=models_torch_range2_physics_01/eui_ann_torch_scalers.joblib
EUI_CORS_ORIGINS=*
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Local development

The original project backend can still be run from the parent project directory:

```bash
uvicorn eui_api:app --reload
```

For the Vercel bundle, install `frontend/requirements.txt` and run from that directory:

```bash
uvicorn app:app --reload
```

## Security

- Keep `.env*` and `.vercel/` out of Git.
- Store the Anthropic key only as a backend environment secret.
- Revoke and replace any key that has appeared in a screenshot, terminal history shared
  publicly, or a Git commit.
- The ANN model artifact does not contain API credentials.
