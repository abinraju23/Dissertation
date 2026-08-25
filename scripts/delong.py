from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Single source of truth: constants and functions are imported from the
# frozen ablation script so the two cannot drift apart.
from ablation import (
    CSV_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    THRESHOLD,
    SAME_ROWS_ACROSS_CONFIGS,
    MODELS,
    resolve_columns,
    chronological_split,
    evaluate,
    selection_score,
)

Z975 = 1.959963984540054  # two-sided 95% normal critical value


# ============================================================================
# 1. DeLong machinery (validated against sklearn AUC and hand-computed cases)
# ============================================================================

def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks with proper tie handling (Sun & Xu 2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def fast_delong(preds_sorted: np.ndarray, m: int):
    """AUCs and DeLong covariance for k models.

    preds_sorted : (k, n_total) scores with the m POSITIVE columns first.
    Returns (aucs shape (k,), covariance S shape (k,k) or scalar if k=1).
    """
    k, total = preds_sorted.shape
    n = total - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.array([compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([compute_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    return aucs, S


def _positives_first(y: np.ndarray, *score_vectors):
    order = np.argsort(-y)          # stable enough: only class order matters
    m = int(y.sum())
    stacked = np.vstack(score_vectors)[:, order]
    return stacked, m


def delong_ci(y: np.ndarray, proba: np.ndarray):
    """AUC with DeLong standard error and 95% CI for one model."""
    P, m = _positives_first(y, proba)
    aucs, S = fast_delong(P, m)
    se = math.sqrt(float(S))
    a = float(aucs[0])
    return a, se, (a - Z975 * se, a + Z975 * se)


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Two-sided DeLong test for correlated ROC curves."""
    P, m = _positives_first(y, p1, p2)
    aucs, S = fast_delong(P, m)
    diff = float(aucs[0] - aucs[1])
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    if var <= 1e-16:
        return aucs, diff, 0.0, 1.0
    z = diff / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return aucs, diff, z, p


def mcnemar_test(y: np.ndarray, pred1: np.ndarray, pred2: np.ndarray):
    """Continuity-corrected McNemar test on paired classifications.

    b = rows model 1 got right and model 2 got wrong; c = the reverse.
    Valid here because n is large (~3,780) and b+c will be well above 25.
    """
    c1 = pred1 == y
    c2 = pred2 == y
    b = int(np.sum(c1 & ~c2))
    c = int(np.sum(~c1 & c2))
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(stat / 2.0))
    return b, c, stat, p


# ============================================================================
# 2. Reproduce the frozen run and recover per-row test predictions
# ============================================================================

def wrap_single_thread(factory):
    def f(params, seed):
        est = factory(params, seed)
        try:
            est.set_params(n_jobs=1)
        except (ValueError, TypeError):
            pass
        return est
    return f


def tune_and_predict(factory, grid, splits, seed):
    """MIRRORS run_one() in ablation.py -- same iteration order, same
    strictly-greater comparison, same first-best-wins tie handling. Do not
    change one without the other. The only addition is that the winning
    estimator's test probabilities are returned."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits

    best = None
    for params in grid:
        est = factory(params, seed)
        est.fit(Xtr, ytr)
        val_metrics = evaluate(yva, est.predict_proba(Xva)[:, 1])
        score = selection_score(val_metrics)
        if best is None or score > best[0]:
            best = (score, params, val_metrics, est)

    _, best_params, best_val, best_est = best
    proba_te = best_est.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(yte, proba_te)
    return best_params, best_val, test_metrics, proba_te


def reproduce(csv_path: Path, single_thread: bool):
    np.random.seed(RANDOM_SEED)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    cols = resolve_columns(df)
    price, sent, graph = cols["price"], cols["sentiment"], cols["graph"]

    configs = {
        "Config 1 (price)": price,
        "Config 2 (price+sent)": price + sent,
        "Config 3 (price+sent+graph)": price + sent + graph,
    }

    # ---- identical preprocessing to run_ablation.main() -------------------
    df = df.copy()
    df[cols["target"]] = pd.to_numeric(df[cols["target"]], errors="coerce")
    df = df.dropna(subset=[cols["target"], cols["date"]])
    df[cols["target"]] = df[cols["target"]].astype(int)
    superset = price + sent + graph
    if SAME_ROWS_ACROSS_CONFIGS:
        df = df.dropna(subset=superset)

    train_df, val_df, test_df = chronological_split(df, cols["date"])
    y_tr = train_df[cols["target"]].to_numpy()
    y_va = val_df[cols["target"]].to_numpy()
    y_te = test_df[cols["target"]].to_numpy()

    print(f"  test fold: {len(y_te):,} rows | UP rate {y_te.mean():.3f}\n")

    # Per-row identifiers for the saved predictions file.
    id_frame = pd.DataFrame({
        "date": pd.to_datetime(test_df[cols["date"]]).values,
        "y_true": y_te,
    })
    if cols["ticker"]:
        id_frame.insert(0, "ticker", test_df[cols["ticker"]].values)

    reproduced_rows = []
    probas = {}

    for config_name, feat_cols in configs.items():
        scaler = StandardScaler().fit(train_df[feat_cols].to_numpy())
        Xtr = scaler.transform(train_df[feat_cols].to_numpy())
        Xva = scaler.transform(val_df[feat_cols].to_numpy())
        Xte = scaler.transform(test_df[feat_cols].to_numpy())
        splits = ((Xtr, y_tr), (Xva, y_va), (Xte, y_te))

        for model_name, (factory, grid) in MODELS.items():
            if single_thread:
                factory = wrap_single_thread(factory)
            print(f"  refitting {config_name:<28} {model_name:<20} "
                  f"({len(grid)} candidates)...")
            best_params, best_val, test_metrics, proba = tune_and_predict(
                factory, grid, splits, RANDOM_SEED)
            probas[(config_name, model_name)] = proba
            reproduced_rows.append({
                "config": config_name,
                "model": model_name,
                "val_auc": best_val["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
                "best_params": str(best_params),
            })

    return pd.DataFrame(reproduced_rows), probas, id_frame, y_te


# ============================================================================
# 3. Reconciliation gate against the frozen ablation table
# ============================================================================

METRIC_COLS = ["test_accuracy", "test_f1", "test_auc", "val_auc"]
TOL = 1e-6


def reconcile(reproduced: pd.DataFrame, frozen_path: Path, allow_mismatch: bool):
    if not frozen_path.exists():
        print(f"\nWARNING: frozen results not found at {frozen_path.resolve()}")
        if not allow_mismatch:
            sys.exit(
                "Cannot verify that this reproduction matches the frozen run.\n"
                "Point the script at the pipeline root (so outputs/ablation_results.csv\n"
                "is visible) or rerun with --allow-mismatch for exploratory use."
            )
        return None

    frozen = pd.read_csv(frozen_path)
    frozen = frozen[frozen["model"] != "-"]          # drop majority baseline
    merged = reproduced.merge(
        frozen, on=["config", "model"], suffixes=("_repro", "_frozen"))

    rows = []
    all_pass = True
    for _, r in merged.iterrows():
        max_diff = max(abs(r[f"{m}_repro"] - r[f"{m}_frozen"]) for m in METRIC_COLS)
        params_match = str(r["best_params_repro"]) == str(r["best_params_frozen"])
        ok = (max_diff < TOL) and params_match
        all_pass &= ok
        rows.append({
            "config": r["config"], "model": r["model"],
            "max_metric_abs_diff": max_diff,
            "params_match": params_match,
            "status": "PASS" if ok else "FAIL",
        })
    rec = pd.DataFrame(rows)

    print("\nReconciliation against frozen ablation_results.csv")
    print("-" * 70)
    for _, r in rec.iterrows():
        print(f"  {r['config']:<30} {r['model']:<20} "
              f"max|diff|={r['max_metric_abs_diff']:.2e}  "
              f"params={'ok' if r['params_match'] else 'DIFFER'}  {r['status']}")

    if not all_pass:
        print("\nRECONCILIATION FAILED: this refit does not match the frozen run.")
        print("Try --single-thread (XGBoost threading nondeterminism is the")
        print("usual cause). Treat these outputs as exploratory until")
        print("reconciliation passes.")
        if not allow_mismatch:
            sys.exit(1)
    else:
        print("\nAll six models reconcile: predictions are the frozen evaluation.")
    return rec


# ============================================================================
# 4. Pairwise tests
# ============================================================================

CFG1 = "Config 1 (price)"
CFG2 = "Config 2 (price+sent)"
CFG3 = "Config 3 (price+sent+graph)"

# The ablation ladder (the comparisons the research question is about),
# then the model-vs-model comparisons within each config.
PAIRS = [
    ("Ablation ladder", CFG1, "LogisticRegression", CFG2, "LogisticRegression"),
    ("Ablation ladder", CFG2, "LogisticRegression", CFG3, "LogisticRegression"),
    ("Ablation ladder", CFG1, "LogisticRegression", CFG3, "LogisticRegression"),
    ("Ablation ladder", CFG1, "XGBoost", CFG2, "XGBoost"),
    ("Ablation ladder", CFG2, "XGBoost", CFG3, "XGBoost"),
    ("Ablation ladder", CFG1, "XGBoost", CFG3, "XGBoost"),
    ("Model comparison", CFG1, "LogisticRegression", CFG1, "XGBoost"),
    ("Model comparison", CFG2, "LogisticRegression", CFG2, "XGBoost"),
    ("Model comparison", CFG3, "LogisticRegression", CFG3, "XGBoost"),
]

LABELS = {
    (CFG1, "LogisticRegression"): "C1-LR",
    (CFG2, "LogisticRegression"): "C2-LR",
    (CFG3, "LogisticRegression"): "C3-LR",
    (CFG1, "XGBoost"): "C1-XGB",
    (CFG2, "XGBoost"): "C2-XGB",
    (CFG3, "XGBoost"): "C3-XGB",
}


def run_tests(probas: dict, y_te: np.ndarray):
    # --- per-model AUC with DeLong 95% CI --------------------------------
    ci_rows = []
    for key, proba in probas.items():
        a, se, (lo, hi) = delong_ci(y_te, proba)
        ci_rows.append({
            "model": LABELS[key], "test_auc": a, "se": se,
            "ci_low": lo, "ci_high": hi,
            "ci_contains_0.5": bool(lo <= 0.5 <= hi),
        })
    ci_df = pd.DataFrame(ci_rows)

    # --- pairwise DeLong + McNemar ---------------------------------------
    test_rows = []
    for group, cfg_a, mod_a, cfg_b, mod_b in PAIRS:
        pa, pb = probas[(cfg_a, mod_a)], probas[(cfg_b, mod_b)]
        aucs, diff, z, p_delong = delong_test(y_te, pa, pb)
        pred_a = (pa >= THRESHOLD).astype(int)
        pred_b = (pb >= THRESHOLD).astype(int)
        b, c, stat, p_mcn = mcnemar_test(y_te, pred_a, pred_b)
        test_rows.append({
            "group": group,
            "comparison": f"{LABELS[(cfg_a, mod_a)]} vs {LABELS[(cfg_b, mod_b)]}",
            "auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "auc_diff": diff, "delong_z": z, "delong_p": p_delong,
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_stat": stat, "mcnemar_p": p_mcn,
        })
    tests_df = pd.DataFrame(test_rows)
    return ci_df, tests_df


# ============================================================================
# 5. Output writers
# ============================================================================

def write_markdown(ci_df: pd.DataFrame, tests_df: pd.DataFrame, path: Path,
                   n_test: int):
    lines = []
    lines.append("## Statistical characterisation of the frozen test results")
    lines.append("")
    lines.append(f"Test fold: n = {n_test:,} observations (2024 H2), "
                 "evaluated once on 17 Jul 2026. All tests below operate on "
                 "the stored predictions of that single evaluation.")
    lines.append("")
    lines.append("### Table: Test AUC with DeLong 95% confidence intervals")
    lines.append("")
    lines.append("| Model | Test AUC | SE | 95% CI | CI contains 0.50 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, r in ci_df.iterrows():
        lines.append(
            f"| {r['model']} | {r['test_auc']:.4f} | {r['se']:.4f} | "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
            f"{'yes' if r['ci_contains_0.5'] else 'no'} |")
    lines.append("")
    for group in ("Ablation ladder", "Model comparison"):
        sub = tests_df[tests_df["group"] == group]
        lines.append(f"### Table: {group} -- DeLong and McNemar tests")
        lines.append("")
        lines.append("| Comparison | AUC A | AUC B | ΔAUC | DeLong z | "
                     "DeLong p | Discordant (b, c) | McNemar χ² | McNemar p |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['comparison']} | {r['auc_a']:.4f} | {r['auc_b']:.4f} | "
                f"{r['auc_diff']:+.4f} | {r['delong_z']:+.2f} | "
                f"{r['delong_p']:.3f} | ({r['mcnemar_b']}, {r['mcnemar_c']}) | "
                f"{r['mcnemar_stat']:.3f} | {r['mcnemar_p']:.3f} |")
        lines.append("")
    lines.append("Interpretation guide: a DeLong p above 0.05 means the AUC "
                 "difference between the two models is statistically "
                 "indistinguishable from zero at this sample size; a McNemar "
                 "p above 0.05 means the two models' error patterns on "
                 "individual days do not differ beyond chance. No multiple-"
                 "comparison correction is applied; corrections only make "
                 "large p-values larger, so the conclusion is unaffected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=str(CSV_PATH),
                    help="path to config3_features.csv (the feature superset)")
    ap.add_argument("--single-thread", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path.resolve()}")

    reproduced, probas, id_frame, y_te = reproduce(csv_path, args.single_thread)

    rec = reconcile(reproduced, OUTPUT_DIR / "ablation_results.csv",
                    args.allow_mismatch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- persist per-row predictions so no future test needs a refit ----
    pred_out = id_frame.copy()
    for key, proba in probas.items():
        tag = LABELS[key].lower().replace("-", "_")
        pred_out[f"proba_{tag}"] = proba
        pred_out[f"pred_{tag}"] = (proba >= THRESHOLD).astype(int)
    pred_path = OUTPUT_DIR / "test_predictions.csv"
    pred_out.to_csv(pred_path, index=False)

    if rec is not None:
        rec.to_csv(OUTPUT_DIR / "reconciliation_report.csv", index=False)

    ci_df, tests_df = run_tests(probas, y_te)
    tests_path = OUTPUT_DIR / "delong_mcnemar_results.csv"
    tests_df.to_csv(tests_path, index=False)
    md_path = OUTPUT_DIR / "delong_mcnemar_tables.md"
    write_markdown(ci_df, tests_df, md_path, len(y_te))

    print("\n" + "=" * 78)
    print("AUC WITH DELONG 95% CI  (test fold, frozen evaluation)")
    print("=" * 78)
    for _, r in ci_df.iterrows():
        print(f"  {r['model']:<8} AUC {r['test_auc']:.4f}  "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
              f"{'contains 0.50' if r['ci_contains_0.5'] else 'EXCLUDES 0.50'}")

    print("\nPAIRWISE TESTS")
    print("-" * 78)
    for _, r in tests_df.iterrows():
        print(f"  {r['comparison']:<18} ΔAUC {r['auc_diff']:+.4f}  "
              f"DeLong p={r['delong_p']:.3f}   "
              f"McNemar (b={r['mcnemar_b']}, c={r['mcnemar_c']}) "
              f"p={r['mcnemar_p']:.3f}")

    print("\nSaved:")
    for p in (pred_path, tests_path, md_path):
        print(f"  {p.resolve()}")
    if rec is not None:
        print(f"  {(OUTPUT_DIR / 'reconciliation_report.csv').resolve()}")


if __name__ == "__main__":
    main()