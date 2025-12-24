import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .inference import (
    ImmunogenicityPredictor,
    align_samples,
    results_to_payload,
)


class PredictRequest(BaseModel):
    heavy_fasta: str = Field(..., description="FASTA-formatted heavy chains")
    light_fasta: Optional[str] = Field(None, description="Optional FASTA-formatted light chains (matching IDs)")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


app = FastAPI(title="Immunogenicity API", version="0.1.0")

_predictor: Optional[ImmunogenicityPredictor] = None


def get_predictor() -> ImmunogenicityPredictor:
    global _predictor
    if _predictor is None:
        model_path = Path("/app/models/marks2021_ensemble.joblib")
        _predictor = ImmunogenicityPredictor(model_path=model_path)
    return _predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    predictor = get_predictor()

    with tempfile.TemporaryDirectory() as tmpdir:
        heavy_path = Path(tmpdir) / "heavy.fa"
        heavy_path.write_text(req.heavy_fasta)

        light_path = None
        if req.light_fasta:
            light_path = Path(tmpdir) / "light.fa"
            light_path.write_text(req.light_fasta)

        samples = align_samples(heavy_path, light_path)
        results = predictor.predict(samples, threshold=req.threshold)

    return {"results": results_to_payload(results)}
