r"""
Statistical characterisation of the null result: DeLong AUC tests and
McNemar tests on the STORED test predictions (checklist item 5 -- pairs
with Agatha's four diagnostics). No model is run; the frozen results are
only read.

What it does
    - DeLong (1988) standard errors and 95% CIs for each config x model AUC,
      plus a z-test of each AUC against chance (0.50). These are the proper
      CIs for the report (they replace the provisional Hanley-McNeil ones
      from diag_roc_curves.py -- expect near-identical numbers).
    - Paired DeLong tests along the ablation ladder, within each model:
        config1 vs config2  (does adding sentiment change AUC?)
        config2 vs config3  (does adding the graph change AUC?)
        config1 vs config3  (do both together change AUC?)
    - McNemar tests on the stored hard predictions for the same pairs
      (do the models make DIFFERENT mistakes, not just equally many?).
    - Holm correction within each family of six pairwise tests.

Run from the project root:   python scripts\diag_delong_mcnemar.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from diag_common import (
    CONFIG_DISPLAY, CONFIG_ORDER, DIAG, MODEL_ORDER, RANDOM_SEED,
    ensure_dirs, load_predictions,
)

N_BOOT = 2000   # date-clustered bootstrap resamples for the vs-chance CIs

PAIRS = [
    ("config1", "config2", "add sentiment"),
    ("config2", "config3", "add graph"),
    ("config1", "config3", "add both"),
]


def date_cluster_bootstrap_ci(y, p, dates, n_boot=N_BOOT, seed=RANDOM_SEED):
    """
    95% CI for the pooled AUC, resampling whole DATES with replacement.
    Same-day rows across the 30 stocks move together, so the plain DeLong
    test (which assumes independent rows) is anti-conservative; resampling
    by date respects that dependence.
    """
    rng = np.random.default_rng(seed)
    dates = np.asarray(dates)
    uniq = np.unique(dates)
    idx_by_date = {d: np.flatnonzero(dates == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_date[d] for d in take])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, p[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------ DeLong (Sun & Xu 2014) --
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _fast_delong(preds_sorted_transposed: np.ndarray, m: int):
    """preds shape (k, n_total), columns sorted so the first m are positives."""
    k, n_total = preds_sorted_transposed.shape
    n = n_total - m
    tx = np.array([_compute_midrank(preds_sorted_transposed[r, :m]) for r in range(k)])
    ty = np.array([_compute_midrank(preds_sorted_transposed[r, m:]) for r in range(k)])
    tz = np.array([_compute_midrank(preds_sorted_transposed[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def _sort_positives_first(y: np.ndarray, probs: list[np.ndarray]):
    order = np.argsort(-y, kind="stable")
    return int(y.sum()), [p[order] for p in probs]


def delong_single(y: np.ndarray, p: np.ndarray):
    """AUC, DeLong SE, 95% CI, z and p against chance (0.50)."""
    m, (ps,) = _sort_positives_first(y, [p])
    aucs, cov = _fast_delong(ps[np.newaxis, :], m)
    auc, se = float(aucs[0]), float(np.sqrt(cov[0, 0]))
    z = (auc - 0.5) / se if se > 0 else 0.0
    pval = 2 * stats.norm.sf(abs(z))
    return auc, se, auc - 1.96 * se, auc + 1.96 * se, z, pval


def delong_paired(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """Paired test of AUC(a) - AUC(b) on the same observations."""
    m, (pa, pb) = _sort_positives_first(y, [p_a, p_b])
    aucs, cov = _fast_delong(np.vstack([pa, pb]), m)
    diff = float(aucs[0] - aucs[1])
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    pval = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), diff, z, pval


# --------------------------------------------------------------- McNemar --
def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray):
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return b, c, np.nan, 1.0, "no discordant pairs"
    if n < 25:
        pval = min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
        return b, c, np.nan, pval, "exact binomial"
    stat = (abs(b - c) - 1) ** 2 / n
    return b, c, stat, stats.chi2.sf(stat, 1), "chi-square (continuity corrected)"


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, order preserved."""
    k = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (k - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ------------------------------------------------------------- alignment --
def model_pivot(preds: pd.DataFrame, mk: str):
    """
    For one model type, return (y, prob_wide, pred_wide, dates) aligned across
    the three configs on identical test observations. dates is None when the
    predictions file has no date column.
    """
    sub = preds[preds["model_key"] == mk].copy()
    keys = [k for k in ("ticker", "date") if k in sub.columns]
    if keys:
        prob = sub.pivot_table(index=keys, columns="config_key", values="y_prob", aggfunc="first")
        ytab = sub.pivot_table(index=keys, columns="config_key", values="y_true", aggfunc="first")
        if "y_pred" in sub.columns:
            pred = sub.pivot_table(index=keys, columns="config_key", values="y_pred", aggfunc="first")
        else:
            pred = (prob >= 0.5).astype(float)
        good = prob.notna().all(axis=1) & pred.notna().all(axis=1)
        prob, pred, ytab = prob[good], pred[good], ytab[good]
        if not (ytab.nunique(axis=1) == 1).all():
            raise SystemExit("[delong] y_true differs across configs for the same observation -- check the predictions file")
        y = ytab.iloc[:, 0].astype(int).to_numpy()
        dates = prob.index.get_level_values("date").to_numpy() if "date" in keys else None
    else:
        blocks_p, blocks_h, y, dates = {}, {}, None, None
        for ck in CONFIG_ORDER:
            g = sub[sub["config_key"] == ck].reset_index(drop=True)
            blocks_p[ck] = g["y_prob"].to_numpy()
            blocks_h[ck] = (g["y_pred"] if "y_pred" in g.columns else (g["y_prob"] >= 0.5)).astype(int).to_numpy()
            if y is None:
                y = g["y_true"].astype(int).to_numpy()
            elif len(g) != len(y) or not np.array_equal(y, g["y_true"].to_numpy()):
                raise SystemExit("[delong] cannot align configs without ticker/date columns -- rows differ")
        prob = pd.DataFrame(blocks_p)
        pred = pd.DataFrame(blocks_h)
    return y, prob, pred, dates


# ------------------------------------------------------------------ main --
def main():
    ensure_dirs()
    preds = load_predictions()
    models = [m for m in MODEL_ORDER if m in preds["model_key"].unique()]

    singles, pairwise, mcn = [], [], []
    for mk in models:
        y, prob, pred, dates = model_pivot(preds, mk)
        n = len(y)

        for ck in CONFIG_ORDER:
            if ck not in prob.columns:
                continue
            p = prob[ck].to_numpy(dtype=float)
            auc, se, lo, hi, z, pval = delong_single(y, p)
            check = roc_auc_score(y, p)
            if abs(check - auc) > 1e-6:
                print(f"[delong] WARNING: internal AUC mismatch for {mk}/{ck} ({auc:.6f} vs {check:.6f})")
            if dates is not None:
                blo, bhi = date_cluster_bootstrap_ci(y, p, dates)
            else:
                blo = bhi = np.nan
            singles.append({"model": mk, "config": ck, "n": n, "auc": auc, "delong_se": se,
                            "ci_low": lo, "ci_high": hi,
                            "boot_ci_low": blo, "boot_ci_high": bhi,
                            "z_vs_chance": z, "p_vs_chance": pval})

        for ck_a, ck_b, what in PAIRS:
            if ck_a not in prob.columns or ck_b not in prob.columns:
                continue
            auc_a, auc_b, diff, z, pval = delong_paired(
                y, prob[ck_a].to_numpy(dtype=float), prob[ck_b].to_numpy(dtype=float))
            pairwise.append({"model": mk, "comparison": f"{ck_a} vs {ck_b}", "change": what,
                             "auc_a": auc_a, "auc_b": auc_b, "auc_diff_b_minus_a": auc_b - auc_a,
                             "z": -z, "p_raw": pval})
            ca = (pred[ck_a].to_numpy(dtype=int) == y)
            cb = (pred[ck_b].to_numpy(dtype=int) == y)
            b, c, stat, pv, method = mcnemar_test(ca, cb)
            mcn.append({"model": mk, "comparison": f"{ck_a} vs {ck_b}", "change": what,
                        "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                        "a_right_b_wrong": b, "a_wrong_b_right": c,
                        "statistic": stat, "p_raw": pv, "method": method})

    df_s = pd.DataFrame(singles)
    df_p = pd.DataFrame(pairwise)
    df_m = pd.DataFrame(mcn)
    if len(df_s):
        df_s["p_holm"] = holm(df_s["p_vs_chance"].tolist())
    if len(df_p):
        df_p["p_holm"] = holm(df_p["p_raw"].tolist())
    if len(df_m):
        df_m["p_holm"] = holm(df_m["p_raw"].tolist())

    for df, name in ((df_s, "delong_auc_single.csv"),
                     (df_p, "delong_pairwise.csv"),
                     (df_m, "mcnemar_pairwise.csv")):
        out = DIAG / name
        df.to_csv(out, index=False)
        print(f"[delong] wrote {out}")

    # ---- console summary ----
    print(f"\n  DeLong AUC vs chance (test set, n = {df_s['n'].iloc[0]} per combo):")
    for _, r in df_s.iterrows():
        boot = (f"  boot [{r.boot_ci_low:.3f}, {r.boot_ci_high:.3f}]"
                if not np.isnan(r.boot_ci_low) else "")
        print(f"    {r.model:<20} {CONFIG_DISPLAY[r.config]:<38} "
              f"AUC = {r.auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]{boot}  "
              f"p = {r.p_vs_chance:.3f}  Holm p = {r.p_holm:.3f}")

    print("\n  paired DeLong -- does adding a feature group change AUC?")
    for _, r in df_p.iterrows():
        print(f"    {r.model:<20} {r.comparison:<22} ({r.change:<13}) "
              f"dAUC = {r.auc_diff_b_minus_a:+.4f}  z = {r.z:+.2f}  "
              f"p = {r.p_raw:.3f}  Holm p = {r.p_holm:.3f}")

    print("\n  McNemar -- do the configs make different mistakes?")
    for _, r in df_m.iterrows():
        print(f"    {r.model:<20} {r.comparison:<22} ({r.change:<13}) "
              f"b = {r.a_right_b_wrong}, c = {r.a_wrong_b_right}  "
              f"p = {r.p_raw:.3f}  Holm p = {r.p_holm:.3f}")

    sig_raw = int((df_s["p_vs_chance"] < 0.05).sum())
    sig_holm_s = int((df_s["p_holm"] < 0.05).sum())
    sig_boot = int(((df_s["boot_ci_low"] > 0.5) | (df_s["boot_ci_high"] < 0.5)).sum()) \
        if df_s["boot_ci_low"].notna().any() else None
    n_sig_pair = int((df_p["p_holm"] < 0.05).sum()) if len(df_p) else 0
    n_sig_mcn = int((df_m["p_holm"] < 0.05).sum()) if len(df_m) else 0

    print(f"\n  summary: vs chance -- {sig_raw}/{len(df_s)} raw p < .05, "
          f"{sig_holm_s}/{len(df_s)} after Holm"
          + (f", {sig_boot}/{len(df_s)} by date-clustered bootstrap CI" if sig_boot is not None else "")
          + ";")
    print(f"           pairwise -- {n_sig_pair}/{len(df_p)} DeLong and "
          f"{n_sig_mcn}/{len(df_m)} McNemar significant after Holm.")
    if sig_raw > 0:
        print("\n  note: the per-observation tests treat all rows as independent, but same-day")
        print("  rows across the 30 stocks are cross-sectionally correlated, so those tests")
        print("  are anti-conservative. Where they disagree, the date-clustered bootstrap CI")
        print("  is the honest interval to report.")

    if sig_holm_s == 0 and n_sig_pair == 0 and n_sig_mcn == 0 and (sig_boot in (0, None)):
        print("\n[delong] done. These three CSVs are the formal backbone of the Results chapter:")
        print("no configuration separably beats chance, and no feature group changes")
        print("performance -- the observed spread is statistically indistinguishable from noise.")
    else:
        print("\n[delong] done. Report the DeLong and bootstrap intervals together and interpret")
        print("any marginal result against the dependence caveat above -- send me the three")
        print("printed blocks and we'll pin the exact wording for the Results chapter.")


if __name__ == "__main__":
    main()