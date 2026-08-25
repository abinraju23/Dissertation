r"""
What it does
    - Reads the STORED test predictions only (no model is run).
    - Draws one panel per model type (Logistic Regression, XGBoost), each
      with the three config curves plus the dashed diagonal for a random
    classifier (AUC = 0.50) as the random-classification baseline.
    - Writes an overall AUC table with 95% CIs (Hanley-McNeil for now; the
      DeLong script will replace/confirm these and add pairwise tests).

Run from the project root:   python scripts\diag_roccurves.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from diag_common import (
    CONFIG_DISPLAY, CONFIG_ORDER, DIAG, FIGURES, MODEL_ORDER,
    auc_ci_hanley, ensure_dirs, load_predictions,
)


def main():
    ensure_dirs()
    preds = load_predictions()

    models = [m for m in MODEL_ORDER if m in preds["model_key"].unique()]
    colors = {"config1": "#4C72B0", "config2": "#DD8452", "config3": "#55A868"}

    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.2), squeeze=False)
    rows = []
    for ax, mk in zip(axes[0], models):
        for ck in CONFIG_ORDER:
            g = preds[(preds["model_key"] == mk) & (preds["config_key"] == ck)]
            if not len(g):
                continue
            y, p = g["y_true"].to_numpy(), g["y_prob"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            lo, hi, _ = auc_ci_hanley(auc, int(y.sum()), int((1 - y).sum()))
            ax.plot(fpr, tpr, color=colors[ck], linewidth=1.6,
                    label=f"{CONFIG_DISPLAY[ck]}  AUC = {auc:.3f}")
            rows.append({"model": mk, "config": ck, "n": len(y),
                         "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi})
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random (AUC = 0.500)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(mk)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")
    fig.suptitle("Test-set ROC curves vs random-chance baseline", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = FIGURES / "roc_curves_test.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[roc] wrote {out_png}")

    table = pd.DataFrame(rows)
    out_csv = DIAG / "overall_auc_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"[roc] wrote {out_csv}\n")
    print("  overall test AUC with 95% CI:")
    for _, r in table.iterrows():
        print(f"    {r.model:<20} {CONFIG_DISPLAY[r.config]:<38} "
              f"AUC = {r.auc:.3f}  [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]  (n = {r.n})")
    print("\n[roc] done. Curves hugging the diagonal with CIs straddling 0.50 is the")
    print("figure that makes the null result visually undeniable in the report.")


if __name__ == "__main__":
    main()