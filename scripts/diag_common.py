r"""
diag_common.py -- shared helpers for the four diagnostic scripts

These scripts only READ stored pipeline outputs (feature CSVs, stored test
predictions, ablation results). None of them run a new evaluation on the
test set or change any frozen result.

Run every script from the project root with the venv active, e.g.:

    python scripts\diag_roccurves.py

If a script cannot find a column it needs, it prints the columns it DID
find and stops. Fill in COLUMN_MAP below only if auto-detection fails.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths --
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
DIAG = OUTPUTS / "diagnostics"

FEATURES_CSV = ROOT / "data" / "config3_features.csv"  # full feature set
PREDICTIONS_CSV = OUTPUTS / "test_predictions.csv"  # stored predictions from the single test run
ABLATION_CSV = OUTPUTS / "ablation_results.csv"     # summary table incl. best_params
MODELS_DIR = ROOT / "models"                        # saved .pkl models, if any exist

# --------------------------------------------------- chronological split --
TRAIN_END = "2023-12-31"   # train:  date <= TRAIN_END
VAL_END = "2024-06-30"     # val:    TRAIN_END < date <= VAL_END; test: after
RANDOM_SEED = 42

# ---------------------------------------------------------- column names --
# Leave values as None to auto-detect. Fill in only if a script tells you to.
COLUMN_MAP = {
    "date": None,
    "ticker": None,
    "label": None,      # 1 = up next day, 0 = down
    "config": None,
    "model": None,
    "y_true": None,
    "y_prob": None,     # predicted probability of class 1 (up)
}

_DATE = ["date", "Date", "datetime", "timestamp"]
_TICKER = ["ticker", "symbol", "Ticker", "Symbol", "stock"]
_LABEL = ["label", "target", "y", "direction", "up_down"]
_CONFIG = ["config", "configuration", "config_name"]
_MODEL = ["model", "model_name", "clf"]
_YTRUE = ["y_true", "y_test", "label", "target", "actual"]
_YPROB = ["y_prob", "y_proba", "prob", "proba", "p_up", "pred_prob", "prob_up", "score"]

# raw price columns that must never be treated as model features
_RAW_PRICE = {"open", "high", "low", "close", "adj close", "adj_close", "volume"}

# name patterns that mark a column as a SENTIMENT feature
SENTIMENT_PATTERNS = ["sent", "finbert", "news", "av_", "alpha"]
# name patterns that mark a column as a GRAPH feature
# (the 64 embedding dims are also caught automatically by the >=32-columns-
#  sharing-a-numeric-suffix-prefix rule in detect_feature_groups)
GRAPH_PATTERNS = ["n2v", "node2vec", "emb", "graph", "degree", "cooc", "co_occ", "pagerank"]

CONFIG_DISPLAY = {
    "config1": "Config 1 (price)",
    "config2": "Config 2 (price + sentiment)",
    "config3": "Config 3 (price + sentiment + graph)",
}
CONFIG_ORDER = ["config1", "config2", "config3"]
MODEL_ORDER = ["LogisticRegression", "XGBoost"]


# ------------------------------------------------------------- utilities --
def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)


_SKIP_DIRS = {"venv", ".venv", "env", ".git", "__pycache__", "node_modules",
              ".ipynb_checkpoints", "figures", "diagnostics"}


def _resolve_file(path: Path, what: str) -> Path:
    """
    Return `path` if it exists; otherwise search the whole project tree for a
    file with the same name (skipping venv etc.). If several copies exist,
    the most recently modified one is used and all candidates are printed.
    """
    if path.exists():
        return path
    matches = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if path.name in files:
            matches.append(Path(root) / path.name)
    if not matches:
        raise SystemExit(
            f"[diag] {what} not found: {path}\n"
            f"Also searched the whole project tree for '{path.name}' with no luck.\n"
            f"Locate it with:  Get-ChildItem -Recurse -Filter {path.name} -Name\n"
            f"then set the path at the top of scripts/diag_common.py."
        )
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(matches) > 1:
        print(f"[diag] found {len(matches)} copies of {path.name}:")
        for m in matches:
            print(f"    {m.relative_to(ROOT)}")
        print("  -> using the most recently modified one")
    chosen = matches[0]
    print(f"[diag] using {what}: {chosen.relative_to(ROOT)}")
    return chosen


def _pick(columns, candidates, override, what, where):
    """Return the first matching column name, honouring a COLUMN_MAP override."""
    if override:
        if override in columns:
            return override
        raise SystemExit(
            f"[diag] COLUMN_MAP['{what}'] = '{override}' is not a column of {where}.\n"
            f"Columns found: {list(columns)}"
        )
    for c in candidates:
        if c in columns:
            return c
    return None


def normalize_config(name: str):
    """Map any config naming ('Config 1 (price)', 'config_1', ...) to config1/2/3."""
    low = str(name).lower()
    if "baseline" in low or "majority" in low:
        return None
    m = re.search(r"[123]", low)
    return f"config{m.group(0)}" if m else None


def normalize_model(name: str):
    low = str(name).lower()
    if "logistic" in low or low in {"lr", "logreg"}:
        return "LogisticRegression"
    if "xgb" in low or "boost" in low:
        return "XGBoost"
    return None


def auc_ci_hanley(auc: float, n1: int, n0: int):
    """95% CI for an AUC via the Hanley-McNeil (1982) standard error."""
    if not (0.0 < auc < 1.0) or min(n1, n0) == 0:
        return np.nan, np.nan, np.nan
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    se = np.sqrt(
        (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2))
        / (n0 * n1)
    )
    return auc - 1.96 * se, auc + 1.96 * se, se


# --------------------------------------------------------------- loaders --
def load_features(path: Path = FEATURES_CSV):
    """Load the full feature CSV. Returns (df, colmap) with parsed dates."""
    path = _resolve_file(path, "features file")
    df = pd.read_csv(path)
    date_col = _pick(df.columns, _DATE, COLUMN_MAP["date"], "date", path.name)
    ticker_col = _pick(df.columns, _TICKER, COLUMN_MAP["ticker"], "ticker", path.name)
    label_col = _pick(df.columns, _LABEL, COLUMN_MAP["label"], "label", path.name)
    missing = [n for n, c in [("date", date_col), ("ticker", ticker_col), ("label", label_col)] if c is None]
    if missing:
        raise SystemExit(
            f"[diag] could not auto-detect {missing} in {path.name}.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Fill in COLUMN_MAP in scripts/diag_common.py and re-run."
        )
    df[date_col] = pd.to_datetime(df[date_col])
    return df, {"date": date_col, "ticker": ticker_col, "label": label_col}


def detect_feature_groups(df: pd.DataFrame, colmap: dict, verbose: bool = True) -> dict:
    """
    Split feature columns into price / sentiment / graph groups by name.

    VERIFY the printed counts against ablation_results.csv n_features:
    expected 16 price, 5 sentiment, 68 graph  ->  configs of 16 / 21 / 89.
    If the counts are wrong, adjust SENTIMENT_PATTERNS / GRAPH_PATTERNS above.
    """
    meta = {colmap["date"], colmap["ticker"], colmap["label"]}
    feature_cols = [
        c for c in df.columns
        if c not in meta
        and c.lower() not in _RAW_PRICE
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # future-looking / target columns stored alongside the features (e.g. the
    # raw next-day return the label is derived from) are NOT model inputs --
    # ablation.py used explicit feature lists that exclude them, so we must too
    future_pats = ["next", "fwd", "future", "tomorrow"]
    excluded = [c for c in feature_cols if any(p in c.lower() for p in future_pats)]
    if excluded:
        print(f"[diag] EXCLUDED from features (target/future-looking, never model inputs): {excluded}")
    feature_cols = [c for c in feature_cols if c not in excluded]

    # embedding block: any prefix followed by digits shared by >= 32 columns
    prefix_groups: dict[str, list] = {}
    for c in feature_cols:
        m = re.match(r"^(.*?)(\d+)$", c)
        if m:
            prefix_groups.setdefault(m.group(1), []).append(c)
    emb_cols = set()
    for cols in prefix_groups.values():
        if len(cols) >= 32:
            emb_cols.update(cols)

    def has(col, patterns):
        low = col.lower()
        return any(p in low for p in patterns)

    graph = [c for c in feature_cols if c in emb_cols or has(c, GRAPH_PATTERNS)]
    graph_set = set(graph)
    sentiment = [c for c in feature_cols if c not in graph_set and has(c, SENTIMENT_PATTERNS)]
    sent_set = set(sentiment)
    price = [c for c in feature_cols if c not in graph_set and c not in sent_set]

    groups = {"price": price, "sentiment": sentiment, "graph": graph}
    if verbose:
        print("[diag] feature group detection -- VERIFY against ablation_results n_features (16 / 21 / 89):")
        print(f"  price     ({len(price):>2}): {price}")
        print(f"  sentiment ({len(sentiment):>2}): {sentiment}")
        preview = graph[:6]
        print(f"  graph     ({len(graph):>2}): {preview}{' ...' if len(graph) > 6 else ''}")
        print(
            f"  -> config1 = {len(price)}, "
            f"config2 = {len(price) + len(sentiment)}, "
            f"config3 = {len(price) + len(sentiment) + len(graph)}"
        )
    return groups


def _try_wide_predictions(df: pd.DataFrame, path_name: str, ticker_col, ytrue_col):
    """
    Detect wide format: one row per (ticker, date) with probability columns
    like proba_c1_lr / proba_c3_xgb, and melt into the long format the
    scripts expect. Matching pred_c*_* columns are carried along as y_pred.
    """
    pat = re.compile(r"^proba?_c(\d+)_([A-Za-z0-9]+)$", re.IGNORECASE)
    prob_cols = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            mk = normalize_model(m.group(2))
            if mk:
                prob_cols[c] = (f"config{m.group(1)}", mk, m.group(1), m.group(2))
    if not prob_cols:
        return None
    if ytrue_col is None:
        raise SystemExit(
            f"[diag] {path_name} has probability columns but no true-label column.\n"
            f"Columns found: {list(df.columns)}"
        )
    date_col = next((c for c in _DATE if c in df.columns), None)
    frames = []
    for c, (ck, mk, knum, msuf) in prob_cols.items():
        keep = [col for col in (ticker_col, date_col, ytrue_col, c) if col]
        f = df[keep].copy()
        rename = {c: "y_prob", ytrue_col: "y_true"}
        if ticker_col:
            rename[ticker_col] = "ticker"
        f = f.rename(columns=rename)
        for cand in (f"pred_c{knum}_{msuf}", c.replace("proba", "pred"), c.replace("prob", "pred")):
            if cand in df.columns:
                f["y_pred"] = df[cand].values
                break
        f["config"] = CONFIG_DISPLAY.get(ck, ck)
        f["model"] = mk
        f["config_key"] = ck
        f["model_key"] = mk
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    configs = sorted({v[0] for v in prob_cols.values()})
    models = sorted({v[1] for v in prob_cols.values()})
    print(f"[diag] {path_name}: wide format detected -- reshaped {len(prob_cols)} "
          f"probability columns into long format ({configs} x {models})")
    return out


def load_predictions(path: Path = PREDICTIONS_CSV) -> pd.DataFrame:
    """
    Load stored test predictions in long format: one row per (config, model,
    observation) with the true label and predicted probability of 'up'.
    Returns a df with standardized columns:
        config, model, config_key, model_key, y_true, y_prob [, ticker]
    """
    path = _resolve_file(path, "predictions file")
    df = pd.read_csv(path)
    cols = df.columns
    config_col = _pick(cols, _CONFIG, COLUMN_MAP["config"], "config", path.name)
    model_col = _pick(cols, _MODEL, COLUMN_MAP["model"], "model", path.name)
    ytrue_col = _pick(cols, _YTRUE, COLUMN_MAP["y_true"], "y_true", path.name)
    yprob_col = _pick(cols, _YPROB, COLUMN_MAP["y_prob"], "y_prob", path.name)
    ticker_col = _pick(cols, _TICKER, COLUMN_MAP["ticker"], "ticker", path.name)

    missing = [n for n, c in [("config", config_col), ("model", model_col),
                              ("y_true", ytrue_col), ("y_prob", yprob_col)] if c is None]
    if missing:
        out = _try_wide_predictions(df, path.name, ticker_col, ytrue_col)
        if out is None:
            raise SystemExit(
                f"[diag] could not auto-detect {missing} in {path.name}.\n"
                f"Columns found: {list(cols)}\n"
                f"Expected long format: one row per prediction with config, model, true label\n"
                f"and predicted probability of class 1. Fill in COLUMN_MAP in diag_common.py,\n"
                f"or paste the printed column list back into the chat and I'll adapt the loader."
            )
    else:
        rename = {config_col: "config", model_col: "model", ytrue_col: "y_true", yprob_col: "y_prob"}
        if ticker_col:
            rename[ticker_col] = "ticker"
        out = df.rename(columns=rename)
        out["config_key"] = out["config"].map(normalize_config)
        out["model_key"] = out["model"].map(normalize_model)

    n_raw = len(out)
    out = out.dropna(subset=["y_prob"])
    out = out[out["config_key"].notna() & out["model_key"].notna()].copy()

    # drop groups with a constant probability (e.g. the majority-class
    # baseline) -- a constant predictor has no ranking, so no ROC/AUC
    nun = out.groupby(["config_key", "model_key"])["y_prob"].transform("nunique")
    dropped = out[nun <= 1][["config", "model"]].drop_duplicates()
    for _, r in dropped.iterrows():
        print(f"[diag] skipping '{r['config']}' / '{r['model']}': constant predicted probability (no ranking)")
    out = out[nun > 1].copy()

    out["y_true"] = out["y_true"].astype(int)
    out["y_prob"] = out["y_prob"].astype(float)
    print(f"[diag] predictions loaded: {len(out)} rows kept of {n_raw} "
          f"({out['config_key'].nunique()} configs x {out['model_key'].nunique()} models"
          f"{', ticker column present' if 'ticker' in out.columns else ', NO ticker column'})")
    return out


def load_best_params(path: Path = ABLATION_CSV) -> dict:
    """Parse best_params per (config_key, model_key) from ablation_results.csv."""
    try:
        path = _resolve_file(path, "ablation results")
    except SystemExit:
        print(f"[diag] note: {path.name} not found anywhere -- refits will use library defaults")
        return {}
    df = pd.read_csv(path)
    config_col = _pick(df.columns, _CONFIG, COLUMN_MAP["config"], "config", path.name)
    model_col = _pick(df.columns, _MODEL, COLUMN_MAP["model"], "model", path.name)
    param_col = next((c for c in df.columns if "param" in c.lower()), None)
    out = {}
    if config_col is None or model_col is None or param_col is None:
        print(f"[diag] note: could not locate config/model/best_params columns in {path.name}")
        return out
    for _, r in df.iterrows():
        ck = normalize_config(r[config_col])
        mk = normalize_model(r[model_col])
        if ck is None or mk is None:
            continue
        try:
            p = ast.literal_eval(str(r[param_col]))
        except (ValueError, SyntaxError):
            p = None
        if isinstance(p, dict):
            out[(ck, mk)] = p
    return out