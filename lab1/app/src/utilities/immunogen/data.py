import numpy as np
import pandas as pd

from src.utilities.immunogen.embeddings import clean_aa_sequence


def add_binary_ada(
    df: pd.DataFrame,
    ada_col: str = "ada_percent",
    threshold: float = 2.0,
    out_col: str = "ada_binary",
) -> pd.DataFrame:
    if ada_col not in df.columns:
        raise ValueError(f"Column '{ada_col}' not found in DataFrame")
    df[out_col] = (df[ada_col] >= threshold).astype(int)
    return df


def add_ada_classes(
    df: pd.DataFrame,
    ada_col: str = "ada_percent",
    thresholds: list[float] = [2.0],
    out_col: str = "ada_class",
    inplace: bool = True,
) -> pd.DataFrame:
    if ada_col not in df.columns:
        raise ValueError(f"Column '{ada_col}' not found in DataFrame")

    thresholds = sorted(thresholds)
    bins = [-np.inf] + thresholds + [np.inf]
    labels = list(range(len(bins) - 1))

    df[out_col] = pd.cut(df[ada_col], bins=bins, labels=labels, right=False).astype(int)

    if inplace:
        df = df.drop(columns=[ada_col])
    return df


def load_marks2021_csv(
    path: str = "src/data/immunogen/marks2021humanization_immunogenicity.csv",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ada_percent" not in df.columns:
        for alt in ["%ADA response", "ADA_percent", "ada", "ADA"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ada_percent"})
                break
    if "ada_percent" not in df.columns:
        raise ValueError("Input CSV must contain 'ada_percent' or an equivalent ADA column.")

    if "heavy" not in df.columns or "light" not in df.columns:
        raise ValueError("Input CSV must contain 'heavy' and 'light' sequence columns.")

    df = df.copy()
    df["heavy"] = df["heavy"].map(clean_aa_sequence)
    df["light"] = df["light"].map(clean_aa_sequence)
    df.insert(0, "antibody_id", np.arange(1, len(df) + 1))
    return df
