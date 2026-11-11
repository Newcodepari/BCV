import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Breast Cancer Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_PATH = "artifacts/pipeline.pkl"
MODEL_PATH = "artifacts/model.pkl"

if not (os.path.exists(PIPELINE_PATH) and os.path.exists(MODEL_PATH)):
    raise FileNotFoundError("Artifacts missing! Train the model first.")

pipeline = joblib.load(PIPELINE_PATH)
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    f"{m}_{f}" for m in ["mean", "se", "worst"] 
    for f in ["radius", "texture", "perimeter", "area", "smoothness",
              "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
]

class BiopsyInput(BaseModel):
    mean_radius: float = 17.99
    mean_texture: float = 10.38
    mean_perimeter: float = 122.80
    mean_area: float = 1001.0
    mean_smoothness: float = 0.11840
    mean_compactness: float = 0.27760
    mean_concavity: float = 0.30010
    mean_concave_points: float = 0.14710
    mean_symmetry: float = 0.2419
    mean_fractal_dimension: float = 0.07871
    se_radius: float = 1.0950
    se_texture: float = 0.9053
    se_perimeter: float = 8.589
    se_area: float = 153.40
    se_smoothness: float = 0.006399
    se_compactness: float = 0.04904
    se_concavity: float = 0.05373
    se_concave_points: float = 0.01587
    se_symmetry: float = 0.03003
    se_fractal_dimension: float = 0.006193
    worst_radius: float = 25.38
    worst_texture: float = 17.33
    worst_perimeter: float = 184.60
    worst_area: float = 2019.0
    worst_smoothness: float = 0.1622
    worst_compactness: float = 0.6656
    worst_concavity: float = 0.7119
    worst_concave_points: float = 0.2654
    worst_symmetry: float = 0.4601
    worst_fractal_dimension: float = 0.11890

@app.get("/")
def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "service": "Breast Cancer Detection System"}

@app.post("/predict")
def predict(data: BiopsyInput):
    try:
        raw_features = np.array([[getattr(data, col) for col in FEATURE_NAMES]])
        scaled_features = pipeline.transform(raw_features)
        
        pred = int(model.predict(scaled_features)[0])
        prob = float(model.predict_proba(scaled_features)[0][1])
        
        return {
            "prediction": pred,
            "diagnosis": "Malignant" if pred == 1 else "Benign",
            "malignant_probability": round(prob, 4),
            "priority_flag": bool(pred == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
