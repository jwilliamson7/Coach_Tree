#!/usr/bin/env python3
"""
Shared train/serve feature pipeline for the play-level XGBoost gene models.

Before this module, all 10 model scripts duplicated categorical encoding, SVD
imputation, and model persistence, and the 4 gene calculators re-implemented the
same encode+impute a fourth way -- which is how the train/serve feature mismatch
(models trained on SVD-reconstructed features, scored on raw median-imputed
features) crept in.

Everything now flows through one place:

  TRAINING:  split_encode_impute()  -> leakage-free split, encoders + imputer fit
             on TRAIN only, returns the fitted estimators
             persist_model()        -> writes model + metadata + encoders + imputer

  SCORING:   load_inference_bundle() -> model + encoders + imputer + metadata
             prepare_features_for_inference() -> transform-only, identical feature
             space to training

Keeping both paths on the same encode/impute code guarantees consistency
structurally instead of by hand. ASCII only (Windows console).
"""

from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


TrainSplit = namedtuple(
    "TrainSplit",
    "X_train X_test y_train y_test feature_names label_encoders imputers",
)

# sentinel category used for missing / unseen categorical values. During training
# (fit=True) NaN becomes the string 'nan' and is learned as a real class, so at
# serve time unseen values map to that same bucket -- a code the model has seen.
_UNSEEN = "nan"


# --------------------------------------------------------------------------- #
# Categorical encoding (shared by training and scoring)
# --------------------------------------------------------------------------- #
def encode_categoricals(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_features: set,
    label_encoders: Dict[str, LabelEncoder],
    fit: bool,
) -> pd.DataFrame:
    """Label-encode categorical columns in place.

    fit=True  : fit a fresh LabelEncoder per categorical column (training).
    fit=False : reuse the encoders in `label_encoders`, mapping unseen values to
                the '_UNSEEN' bucket (scoring / test).

    Any leftover non-numeric columns are dropped (matches prior behavior).
    `label_encoders` is mutated when fit=True.
    """
    df = df.copy()
    categorical_cols = [c for c in feature_names
                        if c in categorical_features and c in df.columns]

    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            if col not in label_encoders:
                continue
            le = label_encoders[col]
            df[col] = df[col].astype(str)
            bucket = _UNSEEN if _UNSEEN in le.classes_ else "unknown"
            if bucket not in le.classes_:
                le.classes_ = np.append(le.classes_, bucket)
            df.loc[~df[col].isin(le.classes_), col] = bucket
            df[col] = le.transform(df[col])

    # drop any remaining non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    if non_numeric:
        df = df.drop(columns=non_numeric)
    return df


# --------------------------------------------------------------------------- #
# SVD imputation (shared by training and scoring)
# --------------------------------------------------------------------------- #
def fit_impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit SimpleImputer(median) + TruncatedSVD on TRAIN only; reconstruct both.

    Returns (X_train_out, X_test_out, estimators) where estimators is
    {'simple', 'svd', 'feature_names'} for later transform-only reuse.
    """
    feature_order = list(X_train.columns)

    simple = SimpleImputer(strategy="median")
    X_train_simple = simple.fit_transform(X_train)
    X_test_simple = simple.transform(X_test)

    n_components = min(50, X_train.shape[1] - 1, X_train.shape[0] - 1)
    svd = None
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_train_out = svd.inverse_transform(svd.fit_transform(X_train_simple))
        X_test_out = svd.inverse_transform(svd.transform(X_test_simple))
    else:
        X_train_out, X_test_out = X_train_simple, X_test_simple

    estimators = {"simple": simple, "svd": svd, "feature_names": feature_order}
    return X_train_out, X_test_out, estimators


def apply_impute(X: pd.DataFrame, estimators: Dict) -> np.ndarray:
    """Transform-only imputation using persisted estimators (scoring path).

    Reindexes X to the training feature order before transforming so the SVD
    components line up with how the model was trained.
    """
    simple = estimators["simple"]
    svd = estimators.get("svd")
    feature_order = estimators.get("feature_names")
    if feature_order is not None:
        X = X.reindex(columns=feature_order)
    X_simple = simple.transform(X)
    if svd is not None:
        return svd.inverse_transform(svd.transform(X_simple))
    return X_simple


# --------------------------------------------------------------------------- #
# Feature selection consumption (parsimony stability-selection output)
# --------------------------------------------------------------------------- #
def apply_feature_selection(feature_names: Sequence[str], model_stem: str) -> List[str]:
    """Intersect the hand-curated feature list with the persisted stability-
    selected set for this model (if any), preserving the input order. Falls back
    to the full list when no selection JSON exists."""
    from utils.parsimony import load_selected_features
    selected = load_selected_features(model_stem)
    if not selected:
        return list(feature_names)
    sel = set(selected)
    return [f for f in feature_names if f in sel]


# --------------------------------------------------------------------------- #
# Training orchestration: the leakage-free split-encode-impute sequence
# --------------------------------------------------------------------------- #
def split_encode_impute(
    feature_data: pd.DataFrame,
    feature_names: Sequence[str],
    target_col: str,
    categorical_features: set,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> TrainSplit:
    """Separate X/y, split, then fit encoders and imputer on TRAIN only.

    This is the single correct ordering used by every model. stratify=True
    (classifiers) stratifies on the integer target; stratify=False (regressors)
    leaves the target continuous.

    Returns a TrainSplit with imputed numeric X_train/X_test arrays, the targets,
    the (possibly reduced) encoded feature_names, and the fitted estimators
    (label_encoders, imputers) for persistence.
    """
    X = feature_data[list(feature_names)].copy()
    y = feature_data[target_col]
    if stratify:
        y = y.astype(int)
    strat = y if stratify else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    label_encoders: Dict[str, LabelEncoder] = {}
    X_train_enc = encode_categoricals(X_train_raw, feature_names, categorical_features,
                                      label_encoders, fit=True)
    X_test_enc = encode_categoricals(X_test_raw, feature_names, categorical_features,
                                     label_encoders, fit=False)

    # encoding may drop non-numeric columns; lock feature order to what survived
    encoded_features = list(X_train_enc.columns)
    X_test_enc = X_test_enc.reindex(columns=encoded_features)

    X_train_imp, X_test_imp, imputers = fit_impute(X_train_enc, X_test_enc)

    return TrainSplit(
        X_train=X_train_imp, X_test=X_test_imp,
        y_train=y_train, y_test=y_test,
        feature_names=encoded_features,
        label_encoders=label_encoders, imputers=imputers,
    )


# --------------------------------------------------------------------------- #
# Scoring orchestration: identical feature space, transform only
# --------------------------------------------------------------------------- #
def prepare_features_for_inference(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_features: set,
    label_encoders: Dict,
    imputers: Optional[Dict],
) -> np.ndarray:
    """Encode (transform-only) then impute (transform-only) so gene-time features
    match training exactly.

    If `imputers` is None (model predates the imputer persistence fix), falls back
    to a fresh median impute and logs nothing here -- callers should warn.
    """
    encoded = encode_categoricals(df, feature_names, categorical_features,
                                  dict(label_encoders or {}), fit=False)
    if imputers is not None:
        return apply_impute(encoded, imputers)
    return SimpleImputer(strategy="median").fit_transform(
        encoded.reindex(columns=list(feature_names)))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def persist_model(
    model,
    filepath: str,
    feature_names: Sequence[str],
    metadata_extra: Optional[Dict] = None,
    label_encoders: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    imputers: Optional[Dict] = None,
    logger=None,
) -> None:
    """Write the XGBoost model (.json), metadata (_metadata.json), encoders
    (_encoders.pkl) and imputer estimators (_imputer.pkl).

    metadata_extra carries model-specific fields (model_type, target_encoding).
    metrics has its non-serializable entries stripped.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_file = f"{filepath}.json"
    model.save_model(model_file)

    metadata = {
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    if hasattr(model, "get_params"):
        try:
            metadata["best_params"] = model.get_params()
        except Exception:
            pass
    if metrics:
        metadata["performance_metrics"] = {
            k: v for k, v in metrics.items()
            if k not in ("predictions_proba", "predictions", "feature_importance")
        }

    with open(f"{filepath}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if label_encoders:
        with open(f"{filepath}_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
    if imputers is not None:
        with open(f"{filepath}_imputer.pkl", "wb") as f:
            pickle.dump(imputers, f)

    if logger:
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metadata saved to {filepath}_metadata.json")
        if label_encoders:
            logger.info(f"Label encoders saved to {filepath}_encoders.pkl")
        if imputers is not None:
            logger.info(f"Imputer estimators saved to {filepath}_imputer.pkl")


def load_inference_bundle(stem: str, model_class) -> Dict:
    """Load model + encoders + imputer + metadata for a model stem.

    model_class is xgb.XGBClassifier or xgb.XGBRegressor. Missing encoders/imputer
    resolve to None (callers fall back / warn). Returns
    {'model','encoders','imputers','metadata','feature_names'}.
    """
    model = model_class()
    model.load_model(f"{stem}.json")

    encoders = None
    enc_path = Path(f"{stem}_encoders.pkl")
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    imputers = None
    imp_path = Path(f"{stem}_imputer.pkl")
    if imp_path.exists():
        with open(imp_path, "rb") as f:
            imputers = pickle.load(f)

    metadata = None
    meta_path = Path(f"{stem}_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    feature_names = metadata.get("feature_names") if metadata else None
    return {"model": model, "encoders": encoders, "imputers": imputers,
            "metadata": metadata, "feature_names": feature_names}
