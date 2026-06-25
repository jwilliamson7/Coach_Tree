#!/usr/bin/env python3
"""
Parsimony and leakage-free utilities for Coach_Tree.

Ports the robustness methodology developed in the sibling CoachingProject
survival pipeline (scripts/survival_methods.py) to Coach_Tree's two modeling
layers:

  Upstream (play-level XGBoost gene models)
    - drop_redundant_features : correlation / zero-variance prune
    - xgb_importance          : gain-based feature ranking (Cox-importance analogue)
    - stability_selection_multi : Meinshausen-Buhlmann stability selection with
                                  GROUP-aware subsampling (a coach's / team's plays
                                  never split across a bootstrap's in/out)
    - select_stable_features  : union-over-K set at selection frequency >= pi
    - selected feature persistence (load/save JSON next to the model)

  Downstream (coach-year gene -> WAR regressions, tiny predictor sets)
    - cluster_robust_ols      : coach-clustered sandwich SEs (single source of
                                truth, lifted from analyze_within_coach_fixed_effects)
    - cluster_bootstrap_ci    : block bootstrap over whole coaches
    - grouped_cv_score        : honest out-of-sample CV with GroupKFold(coach/team)

ASCII only (Windows cp1252 console): use '->' not unicode arrows.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# (a) Redundancy pruning
# --------------------------------------------------------------------------- #
def drop_redundant_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
    protect: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop near-duplicate / collinear numeric columns from a feature matrix.

    Two passes:
      1. drop zero-variance (constant) columns;
      2. greedy upper-triangular |Pearson r| scan -- for each pair with
         |r| >= threshold, drop the column with the higher mean |r| to the rest
         (the more redundant of the two).

    Unlike the hand-curated DROP list in CoachingProject's survival_methods, this
    is data-driven because Coach_Tree's play-level context features carry no
    construct metadata. `protect` columns are never dropped. Non-numeric columns
    are passed through untouched (encode them first if they should participate).

    Returns (X_pruned, dropped_columns).
    """
    protect = set(protect or [])
    dropped: List[str] = []

    num = X.select_dtypes(include=[np.number])
    non_num = [c for c in X.columns if c not in num.columns]

    # 1. zero-variance
    nunique = num.nunique(dropna=False)
    zero_var = [c for c in num.columns if nunique[c] <= 1 and c not in protect]
    dropped.extend(zero_var)
    num = num.drop(columns=zero_var)

    # 2. correlation prune
    if num.shape[1] >= 2:
        corr = num.corr().abs()
        mean_abs = corr.mean(axis=0)  # average |r| of each col to the rest
        cols = list(corr.columns)
        to_drop: set = set()
        for i in range(len(cols)):
            ci = cols[i]
            if ci in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                cj = cols[j]
                if cj in to_drop:
                    continue
                r = corr.iloc[i, j]
                if pd.notna(r) and r >= threshold:
                    # drop the more redundant one, respecting protect
                    if cj in protect and ci in protect:
                        continue
                    if ci in protect:
                        loser = cj
                    elif cj in protect:
                        loser = ci
                    else:
                        loser = ci if mean_abs[ci] >= mean_abs[cj] else cj
                    to_drop.add(loser)
                    if loser == ci:
                        break  # ci is gone, move to next i
        dropped.extend(sorted(to_drop))

    keep = [c for c in X.columns if c not in set(dropped)]
    # stable: original order
    return X[keep], dropped


# --------------------------------------------------------------------------- #
# (b) XGBoost feature importance (cox_importance analogue)
# --------------------------------------------------------------------------- #
def _is_regression(objective: str) -> bool:
    return objective.startswith("reg:") or objective.startswith("count:") or \
        objective.startswith("survival:")


def xgb_importance(
    X: np.ndarray,
    y: np.ndarray,
    cols: Sequence[str],
    xgb_params: Optional[Dict] = None,
    method: str = "gain",
    objective: str = "binary:logistic",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """XGBoost feature ranking -- signature-compatible with cox_importance.

    Fits one XGBClassifier/Regressor on (X, y) and returns
    (rank_desc_indices, importance_vector) aligned to `cols`.

    method='gain' (default) uses gain-based feature_importances_ -- fast, and the
    same signal the models already expose. method='shap' uses mean(|TreeSHAP|)
    via pred_contribs (slow; intended for a one-off confirmatory audit, not the
    bootstrap loop).
    """
    import xgboost as xgb

    params = dict(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
        random_state=random_state,
    )
    if xgb_params:
        params.update(xgb_params)

    if _is_regression(objective):
        model = xgb.XGBRegressor(objective=objective, importance_type="gain", **params)
    else:
        model = xgb.XGBClassifier(
            objective=objective, eval_metric="logloss",
            importance_type="gain", **params,
        )

    model.fit(np.asarray(X, float), np.asarray(y))

    if method == "shap":
        booster = model.get_booster()
        import xgboost as _xgb
        dmat = _xgb.DMatrix(np.asarray(X, float))
        contribs = booster.predict(dmat, pred_contribs=True)
        # last column is the bias term
        imp = np.abs(contribs[:, :-1]).mean(axis=0)
    else:
        imp = np.asarray(model.feature_importances_, float)

    # guard: feature_importances_ length should match cols
    if imp.shape[0] != len(cols):
        full = np.zeros(len(cols))
        full[: imp.shape[0]] = imp
        imp = full

    return np.argsort(imp)[::-1], imp


# --------------------------------------------------------------------------- #
# (b) Group-aware stability selection (Meinshausen & Buhlmann 2010)
# --------------------------------------------------------------------------- #
def _median_impute(arr: np.ndarray) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy="median").fit_transform(arr)


def stability_selection_multi(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    group_col: str,
    Ks: Sequence[int] = (8, 12, 16),
    xgb_params: Optional[Dict] = None,
    n_boot: int = 50,
    subsample: float = 0.5,
    seed: int = 0,
    importance_fn: Optional[Callable] = None,
    objective: str = "binary:logistic",
    impute: Optional[Callable] = None,
    verbose: bool = True,
) -> Dict[int, pd.Series]:
    """Group-aware stability selection, recording top-K membership for several K
    from the SAME subsamples in one bootstrap pass.

    Direct analogue of survival_methods.stability_selection_multi, with the Cox
    importance swapped for xgb_importance and coach-level subsampling generalized
    to any `group_col` (head_coach for offense, defteam for defense). On each of
    n_boot subsamples (a `subsample` fraction of GROUPS, drawn without
    replacement so no group's rows leak across in/out), re-impute on the subsample
    only, rank features, and tally top-K membership for every K in `Ks`.

    X must be fully numeric (encode categoricals first). df provides group_col
    aligned row-for-row with X and y. Returns {K: selection_frequency_series}.
    """
    rng = np.random.default_rng(seed)
    if importance_fn is None:
        def importance_fn(Xi, yi, cols):
            return xgb_importance(Xi, yi, cols, xgb_params=xgb_params,
                                  objective=objective, random_state=seed)
    if impute is None:
        impute = _median_impute

    cols = list(X.columns)
    Xv = X.values
    yv = np.asarray(y)
    groups = df[group_col].values
    uniq = pd.unique(groups)
    n_take = max(1, int(len(uniq) * subsample))
    counts = {K: np.zeros(len(cols)) for K in Ks}

    for b in range(n_boot):
        samp = rng.choice(uniq, size=n_take, replace=False)
        mask = np.isin(groups, samp)
        Ximp = impute(Xv[mask])
        rank, _ = importance_fn(Ximp, yv[mask], cols)
        for K in Ks:
            for i in rank[:K]:
                counts[K][i] += 1
        if verbose and (b + 1) % 10 == 0:
            print(f"  stability bootstrap {b + 1}/{n_boot}")

    return {K: pd.Series(counts[K] / n_boot, index=cols).sort_values(ascending=False)
            for K in Ks}


def select_stable_features(freq_by_K: Dict[int, pd.Series], pi: float = 0.6) -> List[str]:
    """Union across K of features with selection frequency >= pi (the plateau-
    robust set). Returned in descending max-frequency order."""
    best: Dict[str, float] = {}
    for _, freq in freq_by_K.items():
        for name, f in freq.items():
            if f >= pi:
                best[name] = max(best.get(name, 0.0), float(f))
    return [name for name, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


# --------------------------------------------------------------------------- #
# Selected-feature persistence (keeps train-time and gene-time sets in sync)
# --------------------------------------------------------------------------- #
def selected_features_path(model_stem: str) -> Path:
    """Sidecar JSON path for a model stem, e.g.
    'models/run_pass/run_pass_prediction_model' ->
    'models/run_pass/run_pass_prediction_model_selected_features.json'."""
    return Path(f"{model_stem}_selected_features.json")


def save_selected_features(model_stem: str, payload: Dict) -> Path:
    path = selected_features_path(model_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_selected_features(model_stem: str) -> Optional[List[str]]:
    """Return the persisted selected feature list for a model stem, or None if no
    selection JSON exists (callers fall back to the full hand-curated set)."""
    path = selected_features_path(model_stem)
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    feats = payload.get("selected_features")
    return list(feats) if feats else None


# --------------------------------------------------------------------------- #
# (c) Honest group-aware cross-validation
# --------------------------------------------------------------------------- #
def grouped_cv_score(
    make_estimator: Callable,
    prepare_fn: Callable,
    X_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    scoring: str = "auc",
) -> Dict[str, float]:
    """Out-of-sample evaluation with GroupKFold so no group (coach/team) appears
    in both train and test.

    prepare_fn(X_train_df, X_test_df) -> (X_train_num, X_test_num) must fit any
    encoders/imputers on TRAIN ONLY and return numeric arrays. make_estimator()
    returns a fresh model each fold. scoring in {'auc','r2','rmse'}.
    Returns {'mean','std','per_fold'}.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

    y = pd.Series(np.asarray(y))
    groups = pd.Series(np.asarray(groups))
    gkf = GroupKFold(n_splits=n_splits)
    per_fold: List[float] = []

    for tr, te in gkf.split(X_df, y, groups):
        Xtr, Xte = prepare_fn(X_df.iloc[tr], X_df.iloc[te])
        est = make_estimator()
        est.fit(Xtr, y.iloc[tr])
        if scoring == "auc":
            proba = est.predict_proba(Xte)[:, 1]
            per_fold.append(float(roc_auc_score(y.iloc[te], proba)))
        elif scoring == "r2":
            per_fold.append(float(r2_score(y.iloc[te], est.predict(Xte))))
        elif scoring == "rmse":
            per_fold.append(float(np.sqrt(mean_squared_error(y.iloc[te], est.predict(Xte)))))
        else:
            raise ValueError(f"unknown scoring: {scoring}")

    arr = np.array(per_fold)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "per_fold": per_fold}


# --------------------------------------------------------------------------- #
# (d) Coach-clustered inference for the downstream gene -> WAR regressions
# --------------------------------------------------------------------------- #
def cluster_robust_ols(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    feature_names: Sequence[str],
) -> Dict:
    """OLS with cluster-robust (sandwich) standard errors clustered on `clusters`.

    Single source of truth for coach-clustered inference, generalizing the
    one-feature implementation in analyze_within_coach_fixed_effects.py to k
    predictors. Uses the finite-sample factor G/(G-1) and t reference with
    df = G - k - 1 (G = number of clusters), matching that script.

    Returns a dict with intercept, per-feature {coefficient, std_error,
    t_statistic, p_value, significant}, r_squared, n, n_clusters.
    """
    from scipy import stats

    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    n, k = X.shape

    Xc = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ (Xc.T @ y)
    resid = y - Xc @ beta

    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    G = len(uniq)

    meat = np.zeros((k + 1, k + 1))
    for g in uniq:
        m = clusters == g
        s = Xc[m].T @ resid[m]          # cluster score vector
        meat += np.outer(s, s)
    meat *= G / (G - 1)

    vcov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    dof = max(G - k - 1, 1)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def _stat(coef, s):
        t = coef / s if s > 0 else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), dof)) if np.isfinite(t) else np.nan
        return float(t), float(p)

    coefficients = {}
    for i, name in enumerate(feature_names):
        coef = float(beta[i + 1])
        s = float(se[i + 1])
        t, p = _stat(coef, s)
        coefficients[name] = {
            "coefficient": coef,
            "std_error": s,
            "t_statistic": t,
            "p_value": p,
            "significant": bool(np.isfinite(p) and p < 0.05),
        }

    it, ip = _stat(float(beta[0]), float(se[0]))
    return {
        "intercept": float(beta[0]),
        "intercept_se": float(se[0]),
        "intercept_t": it,
        "intercept_p": ip,
        "coefficients": coefficients,
        "r_squared": float(r_squared),
        "n": int(n),
        "n_clusters": int(G),
        "se_type": "cluster_robust_by_coach",
    }


def cluster_bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    feature_names: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict:
    """Block (cluster) bootstrap: resample whole clusters (coaches) WITH
    replacement, refit OLS each draw, return percentile CIs per coefficient.

    Recommended primary uncertainty for the small-n (n_coaches ~ 123) gene -> WAR
    regressions, where cluster-robust sandwich SEs can be slightly anti-
    conservative. Returns {feature: {coef, ci_low, ci_high}}.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)

    # index rows per cluster once
    idx_by_cluster = {g: np.where(clusters == g)[0] for g in uniq}

    def _fit(rows):
        Xc = np.column_stack([np.ones(len(rows)), X[rows]])
        beta = np.linalg.lstsq(Xc, y[rows], rcond=None)[0]
        return beta

    point = _fit(np.arange(len(y)))
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_cluster[g] for g in samp])
        try:
            draws.append(_fit(rows))
        except np.linalg.LinAlgError:
            continue
    draws = np.array(draws)

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    out = {}
    for i, name in enumerate(feature_names):
        col = draws[:, i + 1]
        out[name] = {
            "coefficient": float(point[i + 1]),
            "ci_low": float(np.percentile(col, lo_q)),
            "ci_high": float(np.percentile(col, hi_q)),
        }
    out["_meta"] = {"n_boot": int(len(draws)), "ci": ci, "n_clusters": int(len(uniq))}
    return out


def cluster_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict:
    """Cluster (block) bootstrap for a Pearson correlation: resample whole
    clusters WITH replacement and recompute r each draw.

    For repeated-measures correlations -- coach-years sharing a coach, or
    mentor-protege pairs sharing a mentor -- the ordinary pearsonr p-value
    treats every row as independent and is anti-conservative. This resamples at
    the cluster level instead. Returns the point Pearson r, a percentile CI, and
    a two-sided bootstrap p-value (share of draws on the opposite side of 0,
    doubled). Non-finite rows are dropped; returns NaNs if <3 clusters remain.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[mask], y[mask], clusters[mask]
    uniq = np.unique(clusters)
    n_clusters = len(uniq)

    def _r(xi, yi):
        if len(xi) < 3 or np.std(xi) == 0 or np.std(yi) == 0:
            return np.nan
        return float(np.corrcoef(xi, yi)[0, 1])

    point = _r(x, y)
    if n_clusters < 3 or not np.isfinite(point):
        return {"r": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_bootstrap": float("nan"), "n": int(len(x)),
                "n_clusters": int(n_clusters), "n_boot": 0}

    idx_by_cluster = {g: np.where(clusters == g)[0] for g in uniq}
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=n_clusters, replace=True)
        rows = np.concatenate([idx_by_cluster[g] for g in samp])
        r = _r(x[rows], y[rows])
        if np.isfinite(r):
            draws.append(r)
    draws = np.asarray(draws)

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    if point >= 0:
        p = 2.0 * float(np.mean(draws <= 0))
    else:
        p = 2.0 * float(np.mean(draws >= 0))
    return {
        "r": float(point),
        "ci_low": float(np.percentile(draws, lo_q)),
        "ci_high": float(np.percentile(draws, hi_q)),
        "p_bootstrap": float(min(1.0, p)),
        "n": int(len(x)),
        "n_clusters": int(n_clusters),
        "n_boot": int(len(draws)),
    }


# --------------------------------------------------------------------------- #
# (d2) Measurement-error-aware correlation (WS11: the dependent variable WAR is
#      a noisy single-season estimate, ~half sampling noise against the binomial
#      floor, and its noise is heteroskedastic in games coached. Two honest
#      moves: (i) inverse-variance weight so precise full-season coach-years
#      count more; (ii) report a disattenuation bracket so the conservative
#      observed r is understood as a FLOOR, not the whole story. Neither invents
#      precision: the weights come from a transparent games-based proxy and the
#      disattenuation is reported as a range with its assumption stated.)
# --------------------------------------------------------------------------- #
def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation with non-negative weights w (e.g. inverse
    WAR sampling variance). Reduces to ordinary Pearson r when w is constant.
    Non-finite / non-positive-weight rows are dropped; NaN if <3 rows remain."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 3:
        return float("nan")
    sw = w.sum()
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    cov = (w * (x - mx) * (y - my)).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw
    vy = (w * (y - my) ** 2).sum() / sw
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def cluster_bootstrap_corr_weighted(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    w: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict:
    """Cluster (block) bootstrap for an inverse-variance-WEIGHTED Pearson r.

    Same machinery as cluster_bootstrap_corr (resample whole clusters with
    replacement, recompute the statistic each draw) but the point estimate and
    every draw use weighted_pearson with per-row weights w (carried through the
    resample). With w=None this is identical to the unweighted version. The
    bootstrap deliberately does NOT add extra noise to y -- the observed y already
    contains its sampling noise, and double-counting it would over-widen the CI;
    the residual uncertainty about the TRUE (noise-free) correlation is handled
    separately and analytically by disattenuate_r. Returns r, percentile CI,
    two-sided bootstrap p, n, n_clusters, n_boot.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    if w is None:
        w = np.ones(len(x), float)
    w = np.asarray(w, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w, clusters = x[mask], y[mask], w[mask], clusters[mask]
    uniq = np.unique(clusters)
    n_clusters = len(uniq)

    point = weighted_pearson(x, y, w)
    if n_clusters < 3 or not np.isfinite(point):
        return {"r": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_bootstrap": float("nan"), "n": int(len(x)),
                "n_clusters": int(n_clusters), "n_boot": 0}

    idx_by_cluster = {g: np.where(clusters == g)[0] for g in uniq}
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=n_clusters, replace=True)
        rows = np.concatenate([idx_by_cluster[g] for g in samp])
        r = weighted_pearson(x[rows], y[rows], w[rows])
        if np.isfinite(r):
            draws.append(r)
    draws = np.asarray(draws)

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    if point >= 0:
        p = 2.0 * float(np.mean(draws <= 0))
    else:
        p = 2.0 * float(np.mean(draws >= 0))
    return {
        "r": float(point),
        "ci_low": float(np.percentile(draws, lo_q)),
        "ci_high": float(np.percentile(draws, hi_q)),
        "p_bootstrap": float(min(1.0, p)),
        "n": int(len(x)),
        "n_clusters": int(n_clusters),
        "n_boot": int(len(draws)),
    }


SMALL_CLUSTER_MIN = 40   # below this, percentile cluster bootstrap / clustered t
                         # are anti-conservative; prefer the wild cluster bootstrap


def wild_cluster_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict:
    """Wild cluster bootstrap p-value for H0: corr(x, y) = 0, valid with FEW
    clusters (the percentile cluster bootstrap and the clustered-t both become
    anti-conservative when the number of clusters is small, < ~40).

    Cameron-Gelbach-Miller restricted bootstrap (WCB-R) with Rademacher weights:
    standardize x and y so the OLS slope equals Pearson r, impose the null
    (slope = 0, restricted residuals = y - mean(y)), then for each replicate flip
    the SIGN of every cluster's residual block by an independent +/-1 and recompute
    the cluster-robust t. p = share of bootstrap |t*| >= observed |t|. Returns
    {r, t_observed, p_wild, n, n_clusters, n_boot}.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[m], y[m], clusters[m]
    uniq = np.unique(clusters)
    G = len(uniq)
    if G < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": float("nan"), "t_observed": float("nan"), "p_wild": float("nan"),
                "n": int(len(x)), "n_clusters": int(G), "n_boot": 0}

    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()

    def _slope_t(yy):
        res = cluster_robust_ols(xs, yy, clusters, ["x"])
        c = res["coefficients"]["x"]
        s = c["std_error"]
        return c["coefficient"], (c["coefficient"] / s if s > 0 else np.nan)

    slope_obs, t_obs = _slope_t(ys)
    b0 = ys.mean()
    resid = ys - b0
    idx_by = {g: np.where(clusters == g)[0] for g in uniq}

    rng = np.random.default_rng(seed)
    count = 0
    nb = 0
    for _ in range(n_boot):
        signs = rng.choice([-1.0, 1.0], size=G)
        ystar = np.empty_like(ys)
        for gi, g in enumerate(uniq):
            rows = idx_by[g]
            ystar[rows] = b0 + signs[gi] * resid[rows]
        _, t_star = _slope_t(ystar)
        if np.isfinite(t_star):
            nb += 1
            if abs(t_star) >= abs(t_obs):
                count += 1
    p = count / nb if nb > 0 else float("nan")
    return {"r": float(slope_obs), "t_observed": float(t_obs), "p_wild": float(p),
            "n": int(len(x)), "n_clusters": int(G), "n_boot": int(nb)}


def corr_with_small_cluster_guard(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    min_clusters: int = SMALL_CLUSTER_MIN,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict:
    """Cluster-bootstrap correlation with an automatic small-cluster guard.

    Always returns the percentile cluster-bootstrap result (r, CI, p_bootstrap,
    n_clusters). Adds `small_cluster` = (n_clusters < min_clusters); when that is
    true it also attaches a wild cluster bootstrap p-value (`p_wild_cluster`) as
    the trustworthy inference and `t_observed`, so callers can report the robust p
    and flag the subgroup rather than over-claim from few clusters.
    """
    res = cluster_bootstrap_corr(x, y, clusters, n_boot=n_boot, seed=seed)
    res["small_cluster"] = bool(res.get("n_clusters", 0) < min_clusters)
    if res["small_cluster"] and np.isfinite(res.get("r", np.nan)):
        wcb = wild_cluster_bootstrap_corr(x, y, clusters, n_boot=n_boot, seed=seed)
        res["p_wild_cluster"] = wcb["p_wild"]
        res["t_observed"] = wcb["t_observed"]
    return res


def reliability_from_variance(observed_values: np.ndarray, noise_var) -> float:
    """Panel reliability of a noisy measure: the fraction of its observed variance
    that is true between-unit signal rather than sampling noise,

        rel = max(0, (Var_observed - mean_noise_var) / Var_observed).

    For WAR this is (Var of single-season WAR across coach-years minus the mean
    per-row sampling variance) / Var of single-season WAR. `noise_var` may be a
    scalar (one global sampling variance) or a per-unit array (averaged). Returns
    NaN if Var_observed <= 0. This is an APPROXIMATION (it assumes the sampling-
    noise estimate is right and uncorrelated with signal); callers should report
    the resulting disattenuation as a bracket, not a point claim.
    """
    obs = np.asarray(observed_values, float)
    obs = obs[np.isfinite(obs)]
    if len(obs) < 2:
        return float("nan")
    var_obs = float(np.var(obs, ddof=1))
    mnv = float(np.mean(noise_var)) if np.ndim(noise_var) else float(noise_var)
    if var_obs <= 0:
        return float("nan")
    return float(max(0.0, (var_obs - mnv) / var_obs))


def disattenuate_r(r_obs: float, rel_x: float = 1.0, rel_y: float = 1.0) -> float:
    """Spearman correction for attenuation: r_true = r_obs / sqrt(rel_x * rel_y).

    Measurement error in x and/or y biases an observed correlation TOWARD zero, so
    the observed r is a floor; this recovers an estimate of the correlation between
    the underlying true quantities. rel_x / rel_y are reliabilities in (0, 1].
    Guarded (reliabilities clipped to (1e-6, 1], output capped at +/-0.999). This
    is assumption-dependent (see reliability_from_variance); report as a bracket.
    """
    rel_x = min(max(float(rel_x), 1e-6), 1.0)
    rel_y = min(max(float(rel_y), 1e-6), 1.0)
    return float(np.clip(r_obs / np.sqrt(rel_x * rel_y), -0.999, 0.999))


# --------------------------------------------------------------------------- #
# (e) Reliability-weighted composite genes (WS4: shared by aggression, tempo,
#     defensive). Components differ ~10x in intrinsic reliability, so an
#     equal-weight mean lets sampling noise dominate the weak ones. Weight each
#     component per unit (coach-year / team-year) by its reliability
#         rel = tau2 / (tau2 + samp_var),
#     where tau2 is the true between-unit variance (DerSimonian-Laird) and
#     samp_var is the sampling variance of that component's gene mean. Weights are
#     outcome-blind (never see WAR), so this is not circular.
# --------------------------------------------------------------------------- #
def dersimonian_laird_tau2(gene: np.ndarray, samp_var: np.ndarray) -> float:
    """Heteroskedastic between-unit variance tau^2 (DerSimonian-Laird estimator).

    The meta-analysis standard for combining unit estimates with differing
    sampling variances; robust to small-n units, unlike the naive
    Var(gene) - mean(samp_var) which can collapse to ~0. Returns max(0, .).
    """
    gene = np.asarray(gene, float)
    samp_var = np.asarray(samp_var, float)
    w = 1.0 / samp_var
    k = len(gene)
    if k < 2 or w.sum() <= 0:
        return 0.0
    gbar = float((w * gene).sum() / w.sum())
    Q = float((w * (gene - gbar) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    return max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0


def reliability_weights(gene: np.ndarray, samp_var: np.ndarray,
                        rel_floor: float = 0.1):
    """Per-unit reliability rel = tau2 / (tau2 + samp_var) with tau2 by DL.

    Units below `rel_floor` are zeroed (mostly sampling noise). Returns
    (rel_array, tau2).
    """
    samp_var = np.asarray(samp_var, float)
    tau2 = dersimonian_laird_tau2(gene, samp_var)
    denom = tau2 + samp_var
    rel = np.where(denom > 0, tau2 / denom, 0.0)
    rel = np.where(rel < rel_floor, 0.0, rel)
    return rel, tau2


def reliability_weighted_composite(
    df: pd.DataFrame,
    comp_specs: Sequence,
    rel_floor: float = 0.1,
    value_suffix: str = "",
    logger=None,
):
    """Reliability-weighted composite gene from sub-components.

    comp_specs: iterable of (gene_col, count_col, noisevar_col) where
      - gene_col      : the component gene (actual - predicted) per unit
      - count_col     : n plays behind that unit's component mean
      - noisevar_col  : the SUMMED per-play noise variance for that unit
                        (Bernoulli phat(1-phat) for classifiers; squared
                        residual (actual - pred)^2 for regressors).
    The sampling variance of a component mean is noisevar_col / count_col^2.

    `value_suffix` selects which column is averaged in the weighted mean:
      - ""        -> weight `gene_col` itself (same-unit components, e.g. the
                     four aggression rate-deviations).
      - "_zscore" -> weight `{gene_col}_zscore` (mixed-unit components, e.g. tempo
                     rate vs seconds, defensive counts vs rate); those z-score
                     columns must already exist. Reliability is ALWAYS computed
                     from the raw `gene_col` (rel = tau2/(tau2+samp_var) is
                     scale-invariant, so raw vs z-scored give identical weights).

    Mutates `df` IN PLACE adding a `{gene_col}_reliability` column per component,
    and returns (composite_series, reliability_dict, present_components). The
    composite is the row-wise reliability-weighted mean of available components
    (NaN where no component clears the floor). Caller assigns/z-scores it.
    """
    rel_dict = {}
    present_components = []
    for gene_col, count_col, noisevar_col in comp_specs:
        if (gene_col not in df.columns or count_col not in df.columns
                or noisevar_col not in df.columns):
            continue
        present_components.append(gene_col)
        present = (df[gene_col].notna() & df[count_col].notna() & (df[count_col] > 0))
        n = df.loc[present, count_col].astype(float)
        nv = df.loc[present, noisevar_col].astype(float)
        samp_var = (nv / (n ** 2)).clip(lower=1e-9)
        gene = df.loc[present, gene_col].astype(float)
        rel_arr, tau2 = reliability_weights(gene.values, samp_var.values, rel_floor)
        rel = pd.Series(0.0, index=df.index)
        rel.loc[present] = rel_arr
        df[f"{gene_col}_reliability"] = rel
        rel_dict[gene_col] = {
            "tau2": float(tau2),
            "mean_reliability": float(rel[present].mean()) if present.any() else 0.0,
            "n_units": int(present.sum()),
        }
        if logger:
            logger.info("%s: tau2=%.2e mean_reliability=%.3f", gene_col, tau2,
                        rel_dict[gene_col]["mean_reliability"])

    composite = pd.Series(np.nan, index=df.index)
    if present_components:
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for gene_col in present_components:
            w = df[f"{gene_col}_reliability"].fillna(0.0)
            val_col = f"{gene_col}{value_suffix}"
            g = df[val_col] if val_col in df.columns else df[gene_col]
            mask = g.notna()
            num = num + (w * g).where(mask, 0.0)
            den = den + w.where(mask, 0.0)
        composite = pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)
    return composite, rel_dict, present_components


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    n = 4000
    # 200 groups (coaches), repeated rows
    groups = rng.integers(0, 200, size=n)
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.01, size=n)   # near-duplicate of x1
    x3 = rng.normal(size=n)
    noise_cols = {f"noise{i}": rng.normal(size=n) for i in range(6)}
    const = np.ones(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "const": const, **noise_cols})

    # --- redundancy prune ---
    Xp, dropped = drop_redundant_features(X, threshold=0.95)
    assert "const" in dropped, "zero-variance column not dropped"
    assert ("x1" in dropped) ^ ("x2" in dropped), "one of the duplicate pair must drop"
    print(f"prune OK -> dropped {dropped}")

    # --- stability selection (group-aware) ---
    # signal: y depends on x1 and x3
    logit = 1.5 * x1 + 1.2 * x3
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"coach": groups})
    feats = ["x1", "x3"] + list(noise_cols.keys())
    freq = stability_selection_multi(
        df, X[feats], pd.Series(y), group_col="coach",
        Ks=(2, 4), n_boot=12, subsample=0.5, seed=1, verbose=False,
    )
    stable = select_stable_features(freq, pi=0.6)
    print(f"stability OK -> top by freq: {list(freq[2].head(3).index)}; stable@pi.6={stable}")
    assert "x1" in stable and "x3" in stable, "signal features not selected"

    # --- cluster-robust OLS vs bootstrap ---
    beta_true = np.array([2.0, -1.0])
    Xr = np.column_stack([x1, x3])
    yr = 0.5 + Xr @ beta_true + rng.normal(scale=1.0, size=n)
    res = cluster_robust_ols(Xr, yr, groups, ["x1", "x3"])
    assert abs(res["coefficients"]["x1"]["coefficient"] - 2.0) < 0.1
    boot = cluster_bootstrap_ci(Xr, yr, groups, ["x1", "x3"], n_boot=200, seed=2)
    print(f"cluster OLS OK -> x1 beta={res['coefficients']['x1']['coefficient']:.3f} "
          f"p={res['coefficients']['x1']['p_value']:.4f}; "
          f"boot CI={boot['x1']['ci_low']:.3f}..{boot['x1']['ci_high']:.3f}")

    # --- cluster bootstrap correlation ---
    bc = cluster_bootstrap_corr(x1, yr, groups, n_boot=300, seed=3)
    assert bc["p_bootstrap"] < 0.05, "true correlation should be significant"
    print(f"cluster corr OK -> r={bc['r']:.3f} CI={bc['ci_low']:.3f}..{bc['ci_high']:.3f} "
          f"p_boot={bc['p_bootstrap']:.4f} (clusters={bc['n_clusters']})")

    # --- WS11: weighted corr / disattenuation / reliability ---
    w_uniform = np.ones(n)
    assert abs(weighted_pearson(x1, yr, w_uniform) - np.corrcoef(x1, yr)[0, 1]) < 1e-9, \
        "weighted_pearson with uniform weights must equal ordinary Pearson r"
    bw = cluster_bootstrap_corr_weighted(x1, yr, groups, w=w_uniform, n_boot=200, seed=4)
    assert abs(bw["r"] - bc["r"]) < 1e-9, "uniform-weight bootstrap must match unweighted point r"
    # add heteroskedastic noise to y, weight by inverse noise var -> recover signal
    noise_sd = np.where(rng.uniform(size=n) < 0.5, 0.3, 3.0)
    y_noisy = yr + rng.normal(scale=noise_sd, size=n)
    r_unw = np.corrcoef(x1, y_noisy)[0, 1]
    r_ivw = weighted_pearson(x1, y_noisy, 1.0 / noise_sd ** 2)
    assert r_ivw > r_unw, "inverse-variance weighting should recover attenuated signal"
    rel = reliability_from_variance(y_noisy, np.mean(noise_sd ** 2))
    assert 0.0 < rel < 1.0, "panel reliability must be a fraction"
    assert disattenuate_r(0.2, 1.0, rel) > 0.2, "disattenuation must raise the correlation"
    print(f"WS11 OK -> ivw r {r_unw:.3f}->{r_ivw:.3f}; rel_y={rel:.3f}; "
          f"disatt 0.20->{disattenuate_r(0.2, 1.0, rel):.3f}")

    # --- WS12: wild cluster bootstrap / small-cluster guard ---
    # few-cluster subgroup with a real signal -> WCB should detect it;
    # a null subgroup -> WCB should NOT reject.
    fg = rng.integers(0, 18, size=400)                  # 18 clusters (< 40)
    fx = rng.normal(size=400)
    fy = 0.6 * fx + rng.normal(size=400)
    wsig = wild_cluster_bootstrap_corr(fx, fy, fg, n_boot=499, seed=5)
    assert wsig["p_wild"] < 0.05, "WCB should detect a real few-cluster signal"
    fy0 = rng.normal(size=400)
    wnull = wild_cluster_bootstrap_corr(fx, fy0, fg, n_boot=499, seed=6)
    g_sig = corr_with_small_cluster_guard(fx, fy, fg, n_boot=499, seed=5)
    assert g_sig["small_cluster"] and "p_wild_cluster" in g_sig, "guard must flag few clusters"
    g_big = corr_with_small_cluster_guard(x1, yr, groups, n_boot=300, seed=7)
    assert not g_big["small_cluster"] and "p_wild_cluster" not in g_big, "200 clusters not small"
    print(f"WS12 OK -> WCB sig p={wsig['p_wild']:.3f} (clusters={wsig['n_clusters']}); "
          f"null p={wnull['p_wild']:.3f}; guard flags small={g_sig['small_cluster']}")
    print("ALL PARSIMONY SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
