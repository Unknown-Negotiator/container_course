import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
from Bio import SeqIO

from utilities.immunogen.embeddings import AntiBERTyEmbedder, clean_aa_sequence
from utilities.immunogen.ensemble_model import ImmunogenEnsembleClassifier


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Sample:
    sample_id: str
    heavy: str
    light: Optional[str] = None


@dataclass
class PredictionResult:
    sample_id: str
    prob: float
    cls: int
    sigma: float


def read_fasta(path: Path) -> Dict[str, str]:
    """
    Read FASTA and return id->cleaned sequence.
    Empty or non-AA-only records are dropped.
    """
    records: Dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        seq = clean_aa_sequence(str(record.seq))
        if not seq:
            continue
        records[record.id] = seq
    if not records:
        raise ValueError(f"No usable sequences found in {path}")
    return records


def align_samples(heavy_path: Path, light_path: Optional[Path] = None) -> List[Sample]:
    heavy_map = read_fasta(heavy_path)
    if light_path:
        light_map = read_fasta(light_path)
        missing = set(heavy_map) - set(light_map)
        extra = set(light_map) - set(heavy_map)
        if missing or extra:
            raise ValueError(
                f"Heavy/light FASTA IDs must match. Missing in light: {sorted(missing)}, extra in light: {sorted(extra)}"
            )
        samples = [
            Sample(sample_id=sid, heavy=heavy_map[sid], light=light_map[sid])
            for sid in heavy_map
        ]
    else:
        samples = [Sample(sample_id=sid, heavy=seq) for sid, seq in heavy_map.items()]
    return samples


class ImmunogenicityPredictor:
    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        payload = joblib.load(model_path)
        if not isinstance(payload, dict) or "models" not in payload:
            raise ValueError(f"Unexpected model payload in {model_path}")
        models = payload["models"]
        config = payload.get("config")
        self.model = ImmunogenEnsembleClassifier(models=models, config=config)
        self.embedder = AntiBERTyEmbedder(device=device, batch_size=batch_size)

    def _build_features(self, samples: List[Sample]) -> np.ndarray:
        heavy_seqs = [s.heavy for s in samples]
        light_seqs = [s.light or "" for s in samples]

        heavy_emb = self.embedder.embed_sequences(heavy_seqs)

        # If all light chains are empty, substitute zeros to keep feature size.
        if all(not seq for seq in light_seqs):
            light_emb = np.zeros((len(samples), 512), dtype=np.float32)
        else:
            light_emb = self.embedder.embed_sequences(light_seqs)

        if heavy_emb.shape[0] != light_emb.shape[0]:
            raise RuntimeError("Mismatched heavy/light embeddings")

        return np.hstack([heavy_emb, light_emb]).astype(np.float32)

    def predict(
        self, samples: List[Sample], threshold: float = 0.5
    ) -> List[PredictionResult]:
        if not samples:
            return []
        features = self._build_features(samples)
        mu, sigma = self.model.predict_stats(features)
        results = []
        for sample, prob, stdev in zip(samples, mu, sigma):
            cls = int(prob >= threshold)
            results.append(
                PredictionResult(
                    sample_id=sample.sample_id,
                    prob=float(prob),
                    cls=cls,
                    sigma=float(stdev),
                )
            )
        return results


def results_to_json(results: Iterable[PredictionResult]) -> str:
    payload = results_to_payload(results)
    return json.dumps(payload, indent=2)


def results_to_payload(results: Iterable[PredictionResult]) -> list[dict]:
    return [
        {
            "id": r.sample_id,
            "prob": r.prob,
            "class": r.cls,
            "sigma": r.sigma,
        }
        for r in results
    ]
