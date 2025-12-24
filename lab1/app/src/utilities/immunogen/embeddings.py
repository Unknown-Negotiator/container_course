from typing import List

import numpy as np
import torch
from antiberty import AntiBERTyRunner

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_aa_sequence(s: str) -> str:
    """Uppercase, strip whitespace, keep only canonical amino acids."""
    if s is None:
        return ""
    return "".join([c for c in str(s).upper().strip() if c in AA20])


class AntiBERTyEmbedder:
    def __init__(self, device: str | None = None, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device(self.device)
        self.batch_size = batch_size
        try:
            self.runner = AntiBERTyRunner(device=self.device)
        except TypeError:
            self.runner = AntiBERTyRunner()

    def _pool_embeddings(self, embeddings: List[torch.Tensor]) -> np.ndarray:
        pooled = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            if emb.device.type != self.torch_device.type:
                emb = emb.to(self.torch_device)
            if emb.ndim != 2 or emb.shape[1] != 512:
                raise ValueError(f"Unexpected embedding shape {tuple(emb.shape)}")
            if emb.shape[0] <= 2:
                pooled_vec = emb.mean(dim=0)
            else:
                pooled_vec = emb[1:-1].mean(dim=0)
            pooled.append(pooled_vec.detach().float().cpu().numpy())
        return np.vstack(pooled)

    def embed_sequences(self, seqs: list[str]) -> np.ndarray:
        if not seqs:
            return np.zeros((0, 512), dtype=np.float32)
        cleaned = [clean_aa_sequence(s) for s in seqs]
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(cleaned), self.batch_size):
                batch = cleaned[start : start + self.batch_size]
                embeddings = self.runner.embed(batch)
                pooled = self._pool_embeddings(embeddings)
                outputs.append(pooled.astype(np.float32))
        return np.vstack(outputs)

    def embed_heavy_light_pairs(
        self,
        heavy_seqs: list[str],
        light_seqs: list[str],
    ) -> np.ndarray:
        if len(heavy_seqs) != len(light_seqs):
            raise ValueError("heavy_seqs and light_seqs must have the same length")
        heavy_emb = self.embed_sequences(heavy_seqs)
        light_emb = self.embed_sequences(light_seqs)
        if heavy_emb.shape[0] != light_emb.shape[0]:
            raise RuntimeError("Mismatched embedding counts for heavy and light chains")
        return np.hstack([heavy_emb, light_emb]).astype(np.float32)
