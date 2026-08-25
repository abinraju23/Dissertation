import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "data" / "config3_features.csv")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def flag(cond_ok, msg_ok, msg_flag):
    print(f"  [{'PASS' if cond_ok else 'FLAG'}] {msg_ok if cond_ok else msg_flag}")


# ----------------------------------------------------------------------------
# LOAD + COLUMN RESOLUTION
# ----------------------------------------------------------------------------
hdr("LOAD + COLUMN RESOLUTION")
df = pd.read_csv(CSV)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

cols = df.columns.tolist()
date_col = "date" if "date" in cols else None
tick_col = "ticker" if "ticker" in cols else None
target_col = next((c for c in ["label", "target", "y"] if c in cols), None)
has_retnext = "ret_next" in cols
has_split = "split" in cols

emb_cols = [c for c in cols if re.fullmatch(r"emb_\d+", c)]
cent_kw = ["deg", "central", "clust", "eigen", "between", "pagerank", "graph"]
cent_cols = [c for c in cols
             if any(k in c.lower() for k in cent_kw) and c not in emb_cols]
graph_cols = emb_cols + cent_cols
sent_kw = ["sent", "news", "days_since"]
sent_cols = [c for c in cols
             if any(k in c.lower() for k in sent_kw) and c not in graph_cols]
non_feat = {date_col, tick_col, target_col, "ret_next", "split"}
price_cols = [c for c in cols
              if c not in non_feat and c not in graph_cols and c not in sent_cols]
feature_cols = price_cols + sent_cols + graph_cols

print(f"  rows x cols       : {df.shape[0]:,} x {df.shape[1]}")
print(f"  date / ticker     : {date_col} / {tick_col}")
print(f"  target            : {target_col}")
print(f"  price    features : {len(price_cols):>3}  {price_cols[:8]}{' ...' if len(price_cols) > 8 else ''}")
print(f"  sentiment features: {len(sent_cols):>3}  {sent_cols}")
print(f"  graph    features : {len(graph_cols):>3}  ({len(emb_cols)} emb + {len(cent_cols)} centrality: {cent_cols})")
print(f"  ret_next present  : {has_retnext}   split column present: {has_split}")

if target_col is None:
    sys.exit("  [ABORT] no target column found (looked for label/target/y).")
y_all = df[target_col].astype(int)

# ----------------------------------------------------------------------------
# SPLIT ASSIGNMENT
# ----------------------------------------------------------------------------
if has_split:
    sp = df["split"].astype(str).str.lower()
    tr = sp.str.startswith("train")
    va = sp.str.contains("val")
    te = sp.str.startswith("test")
elif date_col:
    d = df[date_col]
    tr = d <= "2023-12-31"
    va = (d >= "2024-01-01") & (d <= "2024-06-30")
    te = d >= "2024-07-01"
else:
    sys.exit("  [ABORT] no split column and no date column to derive splits.")

# ----------------------------------------------------------------------------
# CHECK 1 -- STRUCTURAL INTEGRITY
# ----------------------------------------------------------------------------
hdr("CHECK 1 -- STRUCTURAL INTEGRITY")
if tick_col:
    per = df.groupby(tick_col).size()
    print(f"  tickers           : {per.shape[0]}")
    print(f"  days per ticker   : min={per.min()} max={per.max()} median={int(per.median())}")
    flag(per.min() == per.max(), "every ticker has the same day count",
         f"uneven day counts (min {per.min()} / max {per.max()}) -- check {per[per != per.median()].index.tolist()[:6]}")
if date_col and tick_col:
    dups = df.duplicated(subset=[date_col, tick_col]).sum()
    flag(dups == 0, "no duplicate (date,ticker) rows", f"{dups} duplicate (date,ticker) rows")
nan_total = int(df[feature_cols].isna().sum().sum())
flag(nan_total == 0, "no NaN in feature matrix",
     f"{nan_total} NaN cells in features (top: {df[feature_cols].isna().sum().sort_values(ascending=False).head(3).to_dict()})")

# ----------------------------------------------------------------------------
# CHECK 2 -- TARGET ALIGNMENT  (bug-vs-real crux, part 1)
# ----------------------------------------------------------------------------
hdr("CHECK 2 -- TARGET ALIGNMENT")
print(f"  UP rate  all/train/val/test : "
      f"{y_all.mean():.3f} / {y_all[tr].mean():.3f} / {y_all[va].mean():.3f} / {y_all[te].mean():.3f}")

if has_retnext:
    rn = df["ret_next"]
    ok = rn.notna()
    agree = (y_all[ok] == (rn[ok] > 0).astype(int)).mean()
    print(f"  agreement( target == (ret_next > 0) ) : {agree:.4f}")
    flag(agree > 0.98, "target matches sign of ret_next -- alignment correct",
         "target does NOT match sign of ret_next -> off-by-one / label bug. "
         "This could produce a chance-level null and should be investigated.")
else:
    print("  ret_next not in CSV -> cannot verify target construction here.")
    print("  >> confirm target[t] = 1 if close[t+1] > close[t] in the build script.")

# same-day leak sniff: today's return should NOT nearly-determine today's target
if "ret_1d" in df.columns:
    c_same = abs(np.corrcoef(df["ret_1d"].fillna(0), y_all)[0, 1])
    flag(c_same < 0.30, f"target not explained by same-day ret_1d (|r|={c_same:.3f})",
         f"same-day ret_1d explains target (|r|={c_same:.3f}) -> target may be aligned to today (leak)")

# ----------------------------------------------------------------------------
# CHECK 3 -- LEAKAGE INTO FEATURES
# ----------------------------------------------------------------------------
hdr("CHECK 3 -- LEAKAGE INTO FEATURES")
flag("ret_next" not in feature_cols, "ret_next excluded from features",
     "ret_next is in the feature set -> direct leakage")
if has_retnext:
    rn = df["ret_next"].fillna(0)
    corrs = {c: abs(np.corrcoef(df[c].fillna(0), rn)[0, 1]) for c in feature_cols}
    worst = sorted(corrs.items(), key=lambda kv: -kv[1])[:5]
    mx = worst[0][1]
    print(f"  max |corr(feature, ret_next)| : {mx:.4f}  ({worst[0][0]})")
    print(f"  top 5             : {[(c, round(v,3)) for c,v in worst]}")
    flag(mx < 0.20, "no feature strongly predicts the future return",
         f"'{worst[0][0]}' predicts future return (|r|={mx:.3f}) -> leak or too-good-to-be-true")

# ----------------------------------------------------------------------------
# CHECK 4 -- DOES SIGNAL EXIST AT ALL  (bug-vs-real crux, part 2)
# ----------------------------------------------------------------------------
hdr("CHECK 4 -- DOES SIGNAL EXIST AT ALL  (train only)")
Xtr_raw = df.loc[tr, feature_cols].apply(pd.to_numeric, errors="coerce")
ytr = y_all[tr].values
med = Xtr_raw.median()
Xtr = Xtr_raw.fillna(med)

# univariate linear association
cc = Xtr.corrwith(pd.Series(ytr, index=Xtr.index)).abs().sort_values(ascending=False)
maxcorr = float(cc.iloc[0])
print("  top 8 |corr(feature, target)| :")
print(cc.head(8).round(4).to_string().replace("\n", "\n    "))

# mutual information (captures non-linear marginal association)
mi = mutual_info_classif(Xtr.values, ytr, discrete_features=False, random_state=0)
mi = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
maxmi = float(mi.iloc[0])
print("\n  top 8 mutual information       :")
print(mi.head(8).round(5).to_string().replace("\n", "\n    "))

# in-sample fit: can a linear model separate the TRAINING data at all?
sc = StandardScaler().fit(Xtr.values)
lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr.values), ytr)
train_auc = roc_auc_score(ytr, lr.predict_proba(sc.transform(Xtr.values))[:, 1])
Xva = df.loc[va, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(med)
yva = y_all[va].values
val_auc = roc_auc_score(yva, lr.predict_proba(sc.transform(Xva.values))[:, 1])
print(f"\n  in-sample train AUC (LogReg)  : {train_auc:.4f}")
print(f"  same model, validation AUC    : {val_auc:.4f}")

# ----------------------------------------------------------------------------
# CHECK 5 -- EMBEDDING HEALTH
# ----------------------------------------------------------------------------
hdr("CHECK 5 -- EMBEDDING HEALTH")
if emb_cols:
    stds = df[emb_cols].std()
    dead = (stds < 1e-6).sum()
    zero_rows = int((df[emb_cols].abs().sum(axis=1) < 1e-9).sum())
    print(f"  emb dims          : {len(emb_cols)}   near-constant dims: {dead}   zero-vector rows: {zero_rows}")
    flag(dead == 0, "no dead embedding dimensions", f"{dead} embedding dims are near-constant")
    if tick_col:
        temporal = df.groupby(tick_col)[emb_cols].std().mean(axis=1)
        cross = df[emb_cols].std().mean()
        print(f"  mean within-ticker temporal std : {temporal.mean():.4f}")
        print(f"  cross-sectional std (overall)   : {cross:.4f}")
        flag(temporal.mean() > 0.05 * cross if cross > 0 else False,
             "embeddings vary over time within a ticker (daily window is doing something)",
             "embeddings are near-static within ticker over time -> graph adds little daily signal")
else:
    print("  no emb_* columns found.")

# ----------------------------------------------------------------------------
# CHECK 6 -- MISSING / COLD-START
# ----------------------------------------------------------------------------
hdr("CHECK 6 -- MISSING / COLD-START")
inf_cnt = int(np.isinf(df[feature_cols].apply(pd.to_numeric, errors="coerce")).sum().sum())
flag(inf_cnt == 0, "no inf values in features", f"{inf_cnt} inf values in features")
if "days_since_news" in df.columns:
    dsn = df["days_since_news"]
    print(f"  days_since_news   : median={dsn.median():.1f}  p90={dsn.quantile(.9):.1f}  max={dsn.max():.1f}")
    print(f"  rows with stale news (>10d) : {(dsn > 10).sum()}")

# ----------------------------------------------------------------------------
# CHECK 7 -- TRAIN -> TEST DRIFT
# ----------------------------------------------------------------------------
hdr("CHECK 7 -- TRAIN -> TEST DRIFT")
probe = [c for c in (["ret_1d"] + sent_cols[:2] + cent_cols[:2]) if c in df.columns]
print(f"  {'feature':<22}{'train mean':>12}{'test mean':>12}{'std diff':>12}")
for c in probe:
    a = pd.to_numeric(df.loc[tr, c], errors="coerce")
    b = pd.to_numeric(df.loc[te, c], errors="coerce")
    pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
    smd = (b.mean() - a.mean()) / pooled
    tag = "  <-- large" if abs(smd) > 0.5 else ""
    print(f"  {c:<22}{a.mean():>12.4f}{b.mean():>12.4f}{smd:>12.2f}{tag}")

# ----------------------------------------------------------------------------
# CHECK 8 -- SPLIT BOUNDARY
# ----------------------------------------------------------------------------
hdr("CHECK 8 -- SPLIT BOUNDARY")
if date_col:
    for name, m in [("train", tr), ("validation", va), ("test", te)]:
        dd = df.loc[m, date_col]
        print(f"  {name:<12} {dd.min().date()} .. {dd.max().date()}   ({m.sum():,} rows, UP={y_all[m].mean():.3f})")
    ordered = df.loc[tr, date_col].max() < df.loc[va, date_col].min() <= df.loc[va, date_col].max() < df.loc[te, date_col].min()
    flag(ordered, "chronological order holds, no overlap", "split dates overlap -> leakage across folds")

# ----------------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------------
hdr("VERDICT")
print(f"  max |corr| feature->target (train) : {maxcorr:.4f}")
print(f"  max mutual information (train)     : {maxmi:.5f}")
print(f"  in-sample train AUC (LogReg)       : {train_auc:.4f}")
print(f"  validation AUC (same model)        : {val_auc:.4f}")
print()

if maxcorr < 0.05 and maxmi < 0.005 and train_auc < 0.56:
    print("  READ: No single feature carries next-day directional signal, AND a linear")
    print("  model cannot fit even the TRAINING data above ~chance. This is what a CORRECT")
    print("  pipeline looks like when the null is genuine. No model class -- XGBoost, LSTM,")
    print("  anything -- recovers signal that is not in the features. The null is real and")
    print("  defensible")
elif train_auc > 0.65 and val_auc < 0.56:
    print("  READ: The model fits TRAIN in-sample but collapses on validation. That is")
    print("  OVERFITTING / distribution mismatch, NOT absence of signal. Worth pursuing:")
    print("  stronger regularisation, cutting the 64-dim embeddings down (PCA / fewer dims")
    print("  for LogReg), and inspecting the train->test drift in Check 7.")
elif maxcorr >= 0.08 or maxmi >= 0.01:
    print("  READ: At least one feature shows non-trivial marginal association with the")
    print("  target, yet the ablation scored at chance. That gap is worth tracing -- check")
    print("  the flagged items above (scaling, feature handling, whether that feature")
    print("  survived into the modelling script).")
else:
    print("  READ: Mixed. Work through any FLAG lines above in order; Checks 2 and 4 decide")
    print("  whether this is a genuine null or a bug suppressing signal.")
print()
print("  Note: any FLAG above must be resolved before the numbers can be trusted.")