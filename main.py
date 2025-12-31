from typing import Dict, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

FEATURE_ORDER = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

CLASS_NAMES = ["malignant", "benign"]

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
LOG_REG_PATH = os.path.join(ARTIFACT_DIR, "log_reg_model.joblib")
TREE_PATH = os.path.join(ARTIFACT_DIR, "decision_tree_model.joblib")


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Mapping from feature name to numeric value"
    )

    def to_frame(self) -> pd.DataFrame:
        missing = [f for f in FEATURE_ORDER if f not in self.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        ordered = {f: self.features[f] for f in FEATURE_ORDER}
        return pd.DataFrame([ordered])


def load_model(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"Model artifact not found at {path}")
    return joblib.load(path)


app = FastAPI(title="Breast Cancer Classifier", version="1.0.0")

# Allow frontend (including local http.server or hosted static site) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}


@app.on_event("startup")
def startup_event():
    global models
    models["log_reg"] = load_model(LOG_REG_PATH)
    models["tree"] = load_model(TREE_PATH)


@app.get("/")
def root():
    return {"status": "ok", "models": list(models.keys()), "features": FEATURE_ORDER}


@app.post("/predict")
def predict(request: PredictRequest, model: Literal["log_reg", "tree"] = "log_reg"):
    if model not in models:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'")
    df = request.to_frame()
    clf = models[model]
    pred = clf.predict(df)[0]
    probs = getattr(clf, "predict_proba", None)
    prob = float(probs(df)[0][pred]) if probs else None
    return {
        "model": model,
        "predicted_class": int(pred),
        "class_name": CLASS_NAMES[pred],
        "probability": prob,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
