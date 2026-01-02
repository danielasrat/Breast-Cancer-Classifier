# ML Assignment: Decision Tree and Logistic Regression

This repo contains a Colab-style notebook for training Decision Tree and Logistic Regression models on the Breast Cancer dataset, saved with joblib, plus a FastAPI backend and a small frontend for inference.

## Project layout

- `ml_pipeline.ipynb` - notebook for data prep, training, evaluation, and artifact export to `artifacts/`.
- `artifacts/` - model binaries saved by the notebook (run the notebook first to populate).
- `main.py` - FastAPI app that serves predictions using the exported models.
- `frontend/index.html` - simple HTML page that calls the backend.
- `requirements.txt` - Python dependencies.

## Quickstart

1. Create env (Python 3.10+ recommended):
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate`
2. Install deps: `pip install -r requirements.txt`
3. Run the notebook to generate artifacts:
   - `jupyter notebook ml_pipeline.ipynb` (or open in VS Code/Colab)
   - Execute all cells; `artifacts/log_reg_model.joblib` and `artifacts/decision_tree_model.joblib` will be written.
4. Start backend: `uvicorn main:app --reload --port 8000`
5. Frontend preview: `cd frontend && python -m http.server 3000` then open http://localhost:3000 and set backend URL to http://localhost:8000/predict.

## API

- `GET /` - model list and feature order.
- `GET /health` - health check.
- `POST /predict?model=log_reg|tree` with body `{ "features": { "mean radius": 14.5, ... } }`.

## Deployment notes

- FastAPI can be deployed on platforms like Render, Railway, Fly.io, or Azure App Service. Use `uvicorn` as the ASGI server. Example Procfile: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Frontend can be deployed on Netlify/Vercel as static hosting; update the backend URL in `frontend/index.html` or via environment config.
- Ensure `artifacts/` is included in the deployment (commit the joblib files or mount persistent storage).
