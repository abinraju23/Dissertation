r"""
What it does
    - Reads the STORED test predictions only (no model is run).
    - Computes AUC, accuracy, and the majority-class share per ticker, for
      every config x model.
    - Puts a 95% CI around each per-ticker AUC (Hanley-McNeil) and counts how
      many tickers sit outside the chance interval -- compared with the ~5%
    expected by chance alone. This assesses whether the null is uniform
    across stocks or driven by a small subset.
    - Draws one dot plot per model type (30 tickers x 3 configs, with the
      0.50 chance line).

Run from the project root:   python scripts\diag_perstock.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from diag_common import (
    CONFIG_DISPLAY, CONFIG_ORDER, DIAG, FIGURES, MODEL_ORDER,
    auc_ci_hanley, ensure_dirs, load_predictions,
)


def main():
    ensure_dirs()
    preds = load_predictions()
    if "ticker" not in preds.columns:
        raise SystemExit(
            "[per-stock] the predictions file has no ticker column, which this "
            "script needs.\nIf the tickers are stored under another name, add it "
            "to COLUMN_MAP['ticker'] in diag_common.py.\nIf they were not stored "
            "at all, verify the prediction file schema and identifier columns."
        )

    rows = []
    for (mk, ck, tkr), g in preds.groupby(["model_key", "config_key", "ticker"]):
        y, p = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
        n1, n0 = int(y.sum()), int((1 - y).sum())
        if min(n1, n0) == 0:
            auc, lo, hi = np.nan, np.nan, np.nan
        else:
            auc = roc_auc_score(y, p)
            lo, hi, _ = auc_ci_hanley(auc, n1, n0)
        if "y_pred" in g.columns and g["y_pred"].notna().all():
            yhat = g["y_pred"].astype(int).to_numpy()   # stored predictions from the frozen run
        else:
            yhat = (p >= 0.5).astype(int)
        acc = float((yhat == y).mean())
        majority = float(max(y.mean(), 1 - y.mean()))
        rows.append({
            "model": mk, "config": ck, "ticker": tkr, "n": len(y),
            "n_up": n1, "n_down": n0,
            "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
            "outside_chance_ci": bool(lo > 0.5 or hi < 0.5) if not np.isnan(auc) else False,
            "accuracy": acc, "majority_share": majority,
            "acc_minus_majority": acc - majority,
        })
    table = pd.DataFrame(rows).sort_values(["model", "config", "ticker"])
    out_csv = DIAG / "per_stock_metrics.csv"
    table.to_csv(out_csv, index=False)
    print(f"[per-stock] wrote {out_csv}")

    # ---- summary ----
    print("\n  per-ticker AUC summary (test period):")
    for mk in MODEL_ORDER:
        for ck in CONFIG_ORDER:
            g = table[(table["model"] == mk) & (table["config"] == ck)]
            if not len(g):
                continue
            n_out = int(g["outside_chance_ci"].sum())
            expected = 0.05 * len(g)
            print(f"    {mk:<20} {ck}:  median {g['auc'].median():.3f}  "
                  f"range [{g['auc'].min():.3f}, {g['auc'].max():.3f}]  "
                  f"outside 95% chance CI: {n_out}/{len(g)} (about {expected:.1f} expected by luck)")
    med_gap = table["acc_minus_majority"].median()
    print(f"\n  median (accuracy - majority-class share) = {med_gap:+.4f}")
    print("  (near zero means accuracy just tracks the class base rate, per ticker too)")

    # ---- one dot plot per model ----
    colors = {"config1": "#4C72B0", "config2": "#DD8452", "config3": "#55A868"}
    for mk in MODEL_ORDER:
        sub = table[table["model"] == mk]
        if not len(sub):
            continue
        order_by = sub[sub["config"] == "config3"].set_index("ticker")["auc"]
        tickers = order_by.sort_values().index.tolist() or sorted(sub["ticker"].unique())
        ypos = {t: i for i, t in enumerate(tickers)}
        fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(tickers) + 1.8))
        for ck in CONFIG_ORDER:
            gg = sub[sub["config"] == ck]
            ax.scatter(gg["auc"], [ypos[t] for t in gg["ticker"]],
                       s=22, label=CONFIG_DISPLAY[ck], color=colors[ck], alpha=0.85)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1,
                   label="Chance (AUC = 0.50)")
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers, fontsize=7)
        ax.set_xlabel("Test AUC")
        ax.set_title(f"Per-stock test AUC -- {mk}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        out = FIGURES / f"per_stock_auc_{mk}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"[per-stock] wrote {out}")

    print("\n[per-stock] done. A tight cloud around 0.50 with ~1-2 tickers outside the CI")
    print("is exactly the 'signal absent everywhere, not hidden in a subset' evidence.")


if __name__ == "__main__":
    main()