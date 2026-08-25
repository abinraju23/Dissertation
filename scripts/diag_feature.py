"""
What it does
    - Recovers the six fitted models (3 configs x 2 model types): loads a
      saved pickle from models/ if one exists, otherwise refits on the
      TRAINING period only, using the best_params stored in
      ablation_results.csv (scaler fit on train only, same as ablation.py).
    - Reads out LR standardized coefficients and XGBoost gain importance.
    - Saves one combined CSV plus six bar charts (top 15 features each).

Test-set integrity
    Nothing here touches the test set. Refitting on training data with the
    hyperparameters already chosen on validation just recovers the fitted
    model objects for readout -- it is not a new design decision and no new
    test evaluation happens.

Run from the project root:   python scripts\diag_feature_importance.py
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diag_common import (
    CONFIG_DISPLAY, DIAG, FIGURES, MODELS_DIR, RANDOM_SEED, TRAIN_END, VAL_END,
    detect_feature_groups, ensure_dirs, load_best_params, load_features,
)

# Set to "train_val" ONLY if ablation.py refit the final models on train+val
# before the single test run. If it evaluated the train-fitted models, keep "train".
REFIT_ON = "train"
TOP_N = 15


def try_load_saved(config_key: str, model_key: str):
    """Look for a saved fitted model, e.g. models/config3_XGBoost.pkl."""
    for name in (f"{config_key}_{model_key}.pkl", f"{model_key}_{config_key}.pkl"):
        p = MODELS_DIR / name
        if p.exists():
            with open(p, "rb") as f:
                obj = pickle.load(f)
            print(f"[importance] loaded saved model {p.name}")
            return obj
    return None


def refit(model_key: str, X: np.ndarray, y: np.ndarray, params: dict):
    if model_key == "LogisticRegression":
        kwargs = dict(max_iter=2000, random_state=RANDOM_SEED)
        kwargs.update(params or {})
        model = LogisticRegression(**kwargs)
    else:
        from xgboost import XGBClassifier
        kwargs = dict(random_state=RANDOM_SEED, eval_metric="logloss")
        kwargs.update(params or {})
        model = XGBClassifier(**kwargs)
    model.fit(X, y)
    return model


def xgb_gain(model, feat_cols):
    """Gain importance mapped back to real feature names (handles f0,f1,... keys)."""
    score = model.get_booster().get_score(importance_type="gain")
    vals = np.zeros(len(feat_cols))
    for key, v in score.items():
        if key in feat_cols:
            vals[feat_cols.index(key)] = v
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(feat_cols):
                vals[idx] = v
    return vals


def plot_top(config_key, model_key, feats, importances, kind):
    order = np.argsort(np.abs(importances))[::-1][:TOP_N][::-1]
    names = [feats[i] for i in order]
    vals = [importances[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.2))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(kind, fontsize=9)
    ax.set_title(f"{CONFIG_DISPLAY[config_key]} -- {model_key}\ntop {len(names)} features", fontsize=10)
    fig.tight_layout()
    out = FIGURES / f"importance_{config_key}_{model_key}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[importance] wrote {out}")


def main():
    ensure_dirs()
    df, colmap = load_features()
    groups = detect_feature_groups(df, colmap)
    config_feats = {
        "config1": groups["price"],
        "config2": groups["price"] + groups["sentiment"],
        "config3": groups["price"] + groups["sentiment"] + groups["graph"],
    }
    best = load_best_params()
    date_col, label_col = colmap["date"], colmap["label"]

    end = pd.Timestamp(VAL_END if REFIT_ON == "train_val" else TRAIN_END)
    fit_mask = df[date_col] <= end
    print(f"[importance] fitting rows: {int(fit_mask.sum())} (dates <= {end.date()}, scaler fit on these only)")

    rows = []
    for config_key, feats in config_feats.items():
        sub = df.loc[fit_mask, feats + [label_col]].dropna()
        X_raw = sub[feats].to_numpy(dtype=float)
        y = sub[label_col].astype(int).to_numpy()
        scaler = StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)

        for model_key in ("LogisticRegression", "XGBoost"):
            model = try_load_saved(config_key, model_key)
            if model is None:
                params = best.get((config_key, model_key), {})
                print(f"[importance] refitting {config_key}/{model_key} "
                      f"({len(feats)} features, params={params or 'library defaults'})")
                model = refit(model_key, X, y, params)

            if model_key == "LogisticRegression":
                imp = np.asarray(model.coef_[0], dtype=float)
                kind = "standardized coefficient"
            else:
                imp = xgb_gain(model, feats)
                kind = "gain"

            for f, v in zip(feats, imp):
                rows.append({
                    "config": config_key, "model": model_key, "feature": f,
                    "importance": v, "abs_importance": abs(v), "importance_type": kind,
                })
            plot_top(config_key, model_key, feats, imp, kind)

    out = pd.DataFrame(rows)
    out_path = DIAG / "feature_importance_all.csv"
    out.to_csv(out_path, index=False)
    print(f"[importance] wrote {out_path}")

    for (ck, mk), g in out.groupby(["config", "model"]):
        top = g.nlargest(8, "abs_importance")
        print(f"\n  top features -- {CONFIG_DISPLAY[ck]} / {mk}:")
        for _, r in top.iterrows():
            print(f"    {r.feature:<30} {r.importance:+.4f}")

    print("\n[importance] done. Expect diffuse importances with no dominant feature --")
    print("the diffuse pattern indicates that no single feature dominates.")


if __name__ == "__main__":
    main()
    