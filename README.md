# Apartment EUI Predictor

An interactive building-energy design tool for a Seattle multifamily prototype. The app
combines a final physics-guided ANN surrogate (`lambda=0.1`) with manual parameter testing,
lower-EUI candidate search, prototype visualization, and Claude-based natural-language
parameter extraction.

## Production

- Website: https://frontend-three-sigma-d9j1jnbhb4.vercel.app
- Health check: https://frontend-three-sigma-d9j1jnbhb4.vercel.app/api/health
- API documentation: https://frontend-three-sigma-d9j1jnbhb4.vercel.app/api/docs

The production frontend and FastAPI backend both run on Vercel. ANN inference uses the
NumPy artifact in `frontend/eui_model_numpy.npz`, exported from the selected PyTorch model.
The older Render deployment is retained only as a fallback and is not called by the Vercel
site during normal use.

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment, model-export, environment-variable,
local-development, and security instructions.
