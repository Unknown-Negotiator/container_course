from typing import List

import numpy as np
import torch
from antiberty import AntiBERTyRunner

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_aa_sequence(s: str) -> str:
    """
    Uppercase, strip whitespace, and keep only the 20 canonical amino acids.
    """
    if s is None:
        return ""
    cleaned = "".join([c for c in str(s).upper().strip() if c in AA20])
    return cleaned


class AntiBERTyEmbedder:
    def __init__(self, device: str | None = None, batch_size: int = 16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device(self.device)
        self.batch_size = batch_size

        runner = None
        try:
            runner = AntiBERTyRunner(device=self.device)
        except TypeError:
            runner = AntiBERTyRunner()
        self.runner = runner

    def _pool_embeddings(self, embeddings: List[torch.Tensor]) -> np.ndarray:
        pooled = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            if not torch.is_tensor(emb):
                raise TypeError(f"Unexpected embedding type: {type(emb)}")
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
        """
        Given a list of aa sequences, compute AntiBERTy per-residue embeddings,
        drop special tokens [0] and [-1], mean-pool over residues,
        and return an array of shape [N, 512].
        """
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
        """
        Clean sequences, compute separate 512-d AntiBERTy embeddings for heavy and light,
        concatenate them into a single 1024-d vector per antibody.
        Returns: np.ndarray of shape [N, 1024].
        """
        if len(heavy_seqs) != len(light_seqs):
            raise ValueError("heavy_seqs and light_seqs must have the same length")
        heavy_emb = self.embed_sequences(heavy_seqs)
        light_emb = self.embed_sequences(light_seqs)
        if heavy_emb.shape[0] != light_emb.shape[0]:
            raise RuntimeError("Mismatched embedding counts for heavy and light chains")
        return np.hstack([heavy_emb, light_emb]).astype(np.float32)
