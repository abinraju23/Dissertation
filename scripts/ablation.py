from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit(
        "xgboost is not installed in this environment.\n"
        "Activate your venv and run:  pip install xgboost"
    )

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Path to the feature superset (Config 1 + 2 + 3 columns in one file).
CSV_PATH = Path("config3_features.csv")

# Where the ablation outputs are written.
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

# Reproducibility.
RANDOM_SEED = 42

# Chronological split boundaries (inclusive).
TRAIN_END = "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-06-30"
TEST_START = "2024-07-01"

# Metric used to pick the best hyperparameters on the validation fold.
# One of: "auc", "f1", "accuracy".
SELECTION_METRIC = "auc"

# Decision threshold for accuracy / F1 (AUC is threshold-free).
THRESHOLD = 0.50

# Evaluate every config on the same rows so differences are attributable to
# features alone. Set to False to let each config use all of its own valid rows.
SAME_ROWS_ACROSS_CONFIGS = True

# Column-name overrides. Leave as None to auto-detect.
DATE_COL: str | None = None
TICKER_COL: str | None = None
TARGET_COL: str | None = None

# Explicitly force columns into a group, or exclude them entirely.
# Anything you list here overrides the auto-detected classification below.
FORCE_PRICE: list[str] = []
FORCE_SENTIMENT: list[str] = []
FORCE_GRAPH: list[str] = []
EXCLUDE_COLS: list[str] = []

# Candidate names for the id / target columns (first match wins).
DATE_CANDIDATES = ["date", "datetime", "timestamp", "trade_date"]
TICKER_CANDIDATES = ["ticker", "symbol", "stock", "sym"]
TARGET_CANDIDATES = ["target", "label", "y", "direction", "next_day_up", "up"]

# Auto-classification rules (matched against lower-cased column names).
GRAPH_PATTERNS = [r"emb[_\-]?\d+", r"node2?vec", r"\bn2v\b"]
GRAPH_KEYWORDS = ["centrality", "pagerank", "betweenness", "closeness",
                  "clustering", "eigen", "degree", "graph", "embed"]
SENTIMENT_KEYWORDS = ["sent", "finbert", "news", "polarity", "decay", "days_since"]

# Any feature whose name looks like it encodes the future is NEVER used, to
# guarantee no target leakage. Matching is broad on purpose: a column named
# e.g. "next_day_return" must be caught. The date/ticker/target columns are
# resolved and removed before this runs, so excluding all "next*"/"future*"
# feature columns is safe. Add to this list if your file has others.
LEAKAGE_PATTERNS = ["next", "future", "forward", "fwd", "tomorrow",
                    "_ahead", "ahead_", "fut_", "_lead", "lead_"]

# -----------------------------------------------------------------------------
# Hyperparameter grids (kept modest for laptop runtime; widen if you have time).
# -----------------------------------------------------------------------------
LR_GRID = [
    {"C": c, "class_weight": cw}
    for c in (0.01, 0.1, 1.0, 10.0)
    for cw in (None, "balanced")
]

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr,
     "subsample": 0.8, "colsample_bytree": 0.8}
    for n in (300, 600)
    for d in (3, 5)
    for lr in (0.03, 0.10)
]


# =============================================================================
# 2. COLUMN RESOLUTION
# =============================================================================

def _first_present(candidates, columns, override):
    if override is not None:
        if override not in columns:
            sys.exit(f"Configured column '{override}' not found in the CSV.")
        return override
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _looks_like_leak(col: str) -> bool:
    lc = col.lower()
    return any(p in lc for p in LEAKAGE_PATTERNS)


def _classify(col: str) -> str:
    """Return 'graph', 'sentiment', or 'price' for a feature column."""
    lc = col.lower()
    if any(re.search(p, lc) for p in GRAPH_PATTERNS) or any(k in lc for k in GRAPH_KEYWORDS):
        return "graph"
    if any(k in lc for k in SENTIMENT_KEYWORDS):
        return "sentiment"
    return "price"


def resolve_columns(df: pd.DataFrame):
    cols = list(df.columns)

    date_col = _first_present(DATE_CANDIDATES, cols, DATE_COL)
    ticker_col = _first_present(TICKER_CANDIDATES, cols, TICKER_COL)
    target_col = _first_present(TARGET_CANDIDATES, cols, TARGET_COL)

    if date_col is None:
        sys.exit("Could not find a date column. Set DATE_COL in the config block.")
    if target_col is None:
        sys.exit("Could not find a target column. Set TARGET_COL in the config block.")

    id_cols = {date_col, ticker_col, target_col} - {None}

    price, sentiment, graph, excluded = [], [], [], []
    forced = {c: "price" for c in FORCE_PRICE}
    forced.update({c: "sentiment" for c in FORCE_SENTIMENT})
    forced.update({c: "graph" for c in FORCE_GRAPH})

    for col in cols:
        if col in id_cols:
            continue
        if col in EXCLUDE_COLS or _looks_like_leak(col):
            excluded.append(col)
            continue
        # Non-numeric columns cannot be model features.
        if not pd.api.types.is_numeric_dtype(df[col]):
            excluded.append(col)
            continue
        group = forced.get(col) or _classify(col)
        {"price": price, "sentiment": sentiment, "graph": graph}[group].append(col)

    return {
        "date": date_col, "ticker": ticker_col, "target": target_col,
        "price": price, "sentiment": sentiment, "graph": graph,
        "excluded": excluded,
    }


# =============================================================================
# 3. SPLITTING & PREP
# =============================================================================

def chronological_split(df, date_col):
    d = pd.to_datetime(df[date_col])
    train = df[d <= pd.Timestamp(TRAIN_END)]
    val = df[(d >= pd.Timestamp(VAL_START)) & (d <= pd.Timestamp(VAL_END))]
    test = df[d >= pd.Timestamp(TEST_START)]

    for name, part in (("train", train), ("validation", val), ("test", test)):
        if part.empty:
            sys.exit(f"The {name} split is empty - check the split dates vs the data range.")

    # Sanity: no temporal overlap.
    assert pd.to_datetime(train[date_col]).max() < pd.to_datetime(val[date_col]).min()
    assert pd.to_datetime(val[date_col]).max() < pd.to_datetime(test[date_col]).min()
    return train, val, test


# =============================================================================
# 4. METRICS
# =============================================================================

def evaluate(y_true, proba, threshold=THRESHOLD):
    pred = (np.asarray(proba) >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": auc,
    }


def selection_score(metrics):
    return metrics[SELECTION_METRIC]


# =============================================================================
# 5. MODEL FACTORIES
# =============================================================================

def make_lr(params, seed):
    return LogisticRegression(
        C=params["C"], class_weight=params["class_weight"],
        solver="liblinear", max_iter=2000, random_state=seed,
    )


def make_xgb(params, seed):
    return XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"], reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0,
    )


MODELS = {
    "LogisticRegression": (make_lr, LR_GRID),
    "XGBoost": (make_xgb, XGB_GRID),
}


# =============================================================================
# 6. TUNE-ON-VALIDATION, SCORE-ON-TEST-ONCE
# =============================================================================

def run_one(model_name, factory, grid, splits, seed):
    """Fit each grid point on TRAIN, pick the best on VALIDATION, then score
    the winning estimator on TEST exactly once."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits

    best = None  # (val_score, params, val_metrics, fitted_estimator)
    for params in grid:
        est = factory(params, seed)
        est.fit(Xtr, ytr)
        val_metrics = evaluate(yva, est.predict_proba(Xva)[:, 1])
        score = selection_score(val_metrics)
        if best is None or score > best[0]:
            best = (score, params, val_metrics, est)

    _, best_params, best_val, best_est = best

    # ---- the single, final touch of the test set -------------------------
    test_metrics = evaluate(yte, best_est.predict_proba(Xte)[:, 1])

    return {
        "model": model_name,
        "best_params": best_params,
        "estimator": best_est,
        "val": best_val,
        "test": test_metrics,
    }


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PATH
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path.resolve()}")

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows x {df.shape[1]} columns\n")

    cols = resolve_columns(df)
    price, sent, graph = cols["price"], cols["sentiment"], cols["graph"]

    print("Column resolution")
    print("-" * 60)
    print(f"  date   : {cols['date']}")
    print(f"  ticker : {cols['ticker']}")
    print(f"  target : {cols['target']}")
    print(f"  price     features ({len(price):>3}): {price[:8]}{' ...' if len(price) > 8 else ''}")
    print(f"  sentiment features ({len(sent):>3}): {sent[:8]}{' ...' if len(sent) > 8 else ''}")
    print(f"  graph     features ({len(graph):>3}): {graph[:6]}{' ...' if len(graph) > 6 else ''}")
    if cols["excluded"]:
        print(f"  excluded  ({len(cols['excluded'])}): {cols['excluded']}")
    print()

    if not price:
        sys.exit("No price features were detected. Check the config block / column names.")

    configs = {
        "Config 1 (price)": price,
        "Config 2 (price+sent)": price + sent,
        "Config 3 (price+sent+graph)": price + sent + graph,
    }

    # ---- target to int, drop rows with missing target --------------------
    df = df.copy()
    df[cols["target"]] = pd.to_numeric(df[cols["target"]], errors="coerce")
    df = df.dropna(subset=[cols["target"], cols["date"]])
    df[cols["target"]] = df[cols["target"]].astype(int)

    # ---- keep identical rows across configs (fair ablation) --------------
    superset = price + sent + graph
    n_before = len(df)
    if SAME_ROWS_ACROSS_CONFIGS:
        df = df.dropna(subset=superset)
        dropped = n_before - len(df)
        if dropped:
            print(f"Dropped {dropped} row(s) with missing feature values "
                  f"({dropped / n_before:.2%}) so all configs share the same rows.\n")

    train_df, val_df, test_df = chronological_split(df, cols["date"])

    def bal(part):
        y = part[cols["target"]]
        return f"{len(y):>6,} rows | UP={y.mean():.3f}"

    print("Chronological split")
    print("-" * 60)
    print(f"  train      (.. {TRAIN_END})      : {bal(train_df)}")
    print(f"  validation ({VAL_START} .. {VAL_END}) : {bal(val_df)}")
    print(f"  test       ({TEST_START} ..)      : {bal(test_df)}")
    print(f"  selection metric on validation      : {SELECTION_METRIC}")
    print(f"  final model fit                     : TRAIN only "
          f"(scaler fit on TRAIN only)\n")

    y_tr = train_df[cols["target"]].to_numpy()
    y_va = val_df[cols["target"]].to_numpy()
    y_te = test_df[cols["target"]].to_numpy()

    rows = []

    # ---- majority-class baseline (informational) -------------------------
    majority = int(round(y_tr.mean()))
    base_pred = np.full_like(y_te, majority)
    rows.append({
        "config": "Baseline (majority class)", "model": "-",
        "n_features": 0,
        "val_auc": np.nan,
        "test_accuracy": accuracy_score(y_te, base_pred),
        "test_f1": f1_score(y_te, base_pred, zero_division=0),
        "test_auc": 0.50,  # a constant predictor has no ranking ability
        "best_params": "predict most frequent train class",
    })

    # ---- every config x every model --------------------------------------
    for config_name, feat_cols in configs.items():
        # Scaler fit on TRAIN ONLY, reused for val and test.
        scaler = StandardScaler().fit(train_df[feat_cols].to_numpy())
        Xtr = scaler.transform(train_df[feat_cols].to_numpy())
        Xva = scaler.transform(val_df[feat_cols].to_numpy())
        Xte = scaler.transform(test_df[feat_cols].to_numpy())
        splits = ((Xtr, y_tr), (Xva, y_va), (Xte, y_te))

        for model_name, (factory, grid) in MODELS.items():
            print(f"  fitting {config_name:<28} {model_name:<20} "
                  f"({len(grid)} candidates on val)...")
            res = run_one(model_name, factory, grid, splits, RANDOM_SEED)
            model_slug = {
                "Config 1 (price)": "config1_price",
                "Config 2 (price+sent)": "config2_price_sentiment",
                "Config 3 (price+sent+graph)": "config3_price_sentiment_graph",
            }[config_name]
            model_path = MODEL_DIR / f"{model_slug}_{model_name.lower()}.joblib"
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": res["estimator"],
                "scaler": scaler,
                "feature_columns": feat_cols,
                "config": config_name,
                "model_name": model_name,
                "best_params": res["best_params"],
                "selection_metric": SELECTION_METRIC,
                "split_dates": {
                    "train_end": TRAIN_END,
                    "validation_start": VAL_START,
                    "validation_end": VAL_END,
                    "test_start": TEST_START,
                },
            }, model_path)
            print(f"    saved model: {model_path}")
            rows.append({
                "config": config_name,
                "model": model_name,
                "n_features": len(feat_cols),
                "val_auc": res["val"]["auc"],
                "test_accuracy": res["test"]["accuracy"],
                "test_f1": res["test"]["f1"],
                "test_auc": res["test"]["auc"],
                "best_params": str(res["best_params"]),
            })

    results = pd.DataFrame(rows)

    # =====================================================================
    # 8. OUTPUT
    # =====================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = OUTPUT_DIR / "ablation_results.csv"
    md_out = OUTPUT_DIR / "ablation_table.md"

    results.to_csv(csv_out, index=False)
    _write_markdown(results, md_out)

    print("\n" + "=" * 78)
    print("ABLATION RESULTS  (test fold, scored once)")
    print("=" * 78)
    _print_table(results)
    print("\nSaved:")
    print(f"  {csv_out.resolve()}")
    print(f"  {md_out.resolve()}")


# -----------------------------------------------------------------------------
# small self-contained table writers (no tabulate dependency)
# -----------------------------------------------------------------------------
DISPLAY_COLS = [
    ("config", "Configuration", 30, "s"),
    ("model", "Model", 20, "s"),
    ("n_features", "Feat", 5, "d"),
    ("val_auc", "Val AUC", 9, "f"),
    ("test_accuracy", "Test Acc", 9, "f"),
    ("test_f1", "Test F1", 9, "f"),
    ("test_auc", "Test AUC", 9, "f"),
]


def _fmt(val, kind):
    if kind == "f":
        return "   -   " if pd.isna(val) else f"{val:.4f}"
    if kind == "d":
        return f"{int(val)}"
    return str(val)


def _print_table(df):
    header = "  ".join(f"{title:<{w}}" for _, title, w, _ in DISPLAY_COLS)
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        line = "  ".join(f"{_fmt(r[key], kind):<{w}}" for key, _, w, kind in DISPLAY_COLS)
        print(line)


def _write_markdown(df, path):
    heads = [title for _, title, _, _ in DISPLAY_COLS]
    lines = ["| " + " | ".join(heads) + " |",
             "| " + " | ".join("---" for _ in heads) + " |"]
    for _, r in df.iterrows():
        cells = [_fmt(r[key], kind).strip() for key, _, _, kind in DISPLAY_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()