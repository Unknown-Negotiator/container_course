from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ImmunogenEnsembleConfig:
    threshold: float = 2.0
    n_models: int = 5
    random_state: int = 42
    cv_splits: int = 5
    pca_components_grid: List[int] | None = None
    c_grid: List[float] | None = None


class ImmunogenEnsembleClassifier:
    def __init__(self, models: List[Pipeline], config: ImmunogenEnsembleConfig):
        self.models = models
        self.config = config

    @classmethod
    def train_from_embeddings_df(
        cls,
        df: pd.DataFrame,
        config: ImmunogenEnsembleConfig | None = None,
    ) -> "ImmunogenEnsembleClassifier":
        from src.utilities.immunogen.data import add_binary_ada

        config = config or ImmunogenEnsembleConfig()
        df = df.copy()
        df = add_binary_ada(df, threshold=config.threshold)

        feature_cols = [
            c for c in df.columns if c.startswith("heavy_") or c.startswith("light_")
        ]
        if not feature_cols:
            raise RuntimeError("No embedding feature columns found.")

        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["ada_binary"].to_numpy(dtype=int)

        n_samples, n_features = X.shape
        pca_max = max(1, min(n_samples - 1, n_features))

        if config.pca_components_grid is None:
            pca_grid = [10, 25, 50, 75, 100]
        else:
            pca_grid = list(config.pca_components_grid)
        pca_grid = sorted({int(c) for c in pca_grid if 1 <= int(c) <= pca_max})
        if not pca_grid:
            pca_grid = [min(10, pca_max)]

        if config.c_grid is None:
            c_grid = list(np.logspace(-3, 3, 7))
        else:
            c_grid = list(config.c_grid)
        c_grid = [float(c) for c in c_grid if c > 0]
        if not c_grid:
            c_grid = [1.0]

        models: List[Pipeline] = []
        for i in range(config.n_models):
            rng = np.random.default_rng(config.random_state + i)
            indices = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            pipe = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("pca", PCA(random_state=config.random_state + i)),
                    ("logreg", LogisticRegression(max_iter=1000, solver="lbfgs")),
                ]
            )
            param_grid = {
                "pca__n_components": pca_grid,
                "logreg__C": c_grid,
            }
            cv = StratifiedKFold(
                n_splits=config.cv_splits,
                shuffle=True,
                random_state=config.random_state + i,
            )
            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_boot, y_boot)
            models.append(search.best_estimator_)

        return cls(models=models, config=config)

    def predict_proba_members(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for model in self.models:
            member_probs = model.predict_proba(X)[:, 1]
            probs.append(member_probs)
        return np.stack(probs, axis=1)

    def predict_stats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        member_probs = self.predict_proba_members(X)
        mu = member_probs.mean(axis=1)
        sigma = member_probs.std(axis=1)
        return mu, sigma

    def save(self, path: str) -> None:
        joblib.dump({"models": self.models, "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "ImmunogenEnsembleClassifier":
        payload = joblib.load(path)
        models = payload["models"]
        config = payload["config"]
        return cls(models=models, config=config)
