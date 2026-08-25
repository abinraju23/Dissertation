# Leveraging Financial News, NLP & Knowledge Graphs for Stock Direction Prediction

> **Master's Dissertation Research Pipeline**
> A reproducible machine-learning pipeline for evaluating whether financial-news sentiment and news co-occurrence knowledge graphs improve next-day stock-price direction prediction.

---

## 📌 Overview

This project investigates whether **financial news sentiment** and **relational information captured through knowledge graphs** can improve short-term stock price direction classification.

The pipeline builds a reproducible panel covering **30 US-listed companies**, creates three nested feature configurations, trains two classification models, and evaluates them using a strict **chronological train/validation/test design**.

### Research Progression

| Configuration | Feature Block | Research Purpose |
|:---:|---|---|
| **Config 1** | Price-derived features | Establish the technical baseline |
| **Config 2** | Config 1 + FinBERT + news features | Measure the incremental contribution of financial-news sentiment |
| **Config 3** | Config 2 + Node2Vec graph features | Measure the contribution of relational/news-network structure |

### 🎯 Prediction Target

For each stock on trading day `t`:

- **Target = 1** → next day's closing price is higher than today's close
- **Target = 0** → next day's closing price is not higher than today's close

### ⏱️ Chronological Evaluation

| Dataset Split | Period | Purpose |
|---|---|---|
| **Training** | 2022–2023 | Model fitting and parameter tuning |
| **Validation** | 2024 H1 | Hyperparameter/model selection |
| **Final Test** | 2024 H2 | Unseen final evaluation |

> **No future test-period information is used for hyperparameter selection.**

---

## 🧠 Methodology

The pipeline follows the research flow below:

```text
Financial News ───────┐
                      │
                      ▼
                 News Processing
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     FinBERT NLP            Entity/News Links
          │                       │
          ▼                       ▼
   Sentiment Features      Co-occurrence Graph
                                  │
                                  ▼
                              Node2Vec
                                  │
                                  ▼
                          Graph Embeddings
                                  │
                                  └──────┐
                                         ▼
Historical Prices ──► Price Features ─► Feature Configurations
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
                 Config 1             Config 2             Config 3
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                              Logistic Regression
                                      +
                                   XGBoost
                                         │
                                         ▼
                             Chronological Evaluation
                                         │
                                         ▼
                           Statistical Tests & Diagnostics
```

---

## 📁 Project Structure

```text
project-root/
│
├── data/
│   ├── raw/news/              # Original Alpha Vantage responses
│   ├── prices/                # Per-stock price data
│   ├── finbert_cache.csv      # Cached FinBERT scores
│   ├── graph_emb_cache.pkl    # Cached graph embeddings
│   ├── config1_features.csv   # Price-only features
│   ├── config2_features.csv   # Price + sentiment/news features
│   └── config3_features.csv   # Price + sentiment + graph features
│
├── figures/                   # General research figures
│
├── models/                    # Saved model/scaler bundles
│
├── outputs/                   # Predictions, diagnostics & statistical results
│
├── scripts/
│   ├── newscoverage.py
│   ├── collectnews.py
│   ├── build_price.py
│   ├── build_sentiment.py
│   ├── build_graph.py
│   ├── ablation.py
│   ├── delong.py
│   └── diagnostic scripts
│
├── run_pipeline.py            # Ordered pipeline runner
├── requirements.txt           # Python dependencies
└── README.md
```

> The original Alpha Vantage responses are retained under `data/raw/news/`, allowing later feature-engineering stages to be inspected and rerun.

---

# ⚙️ Environment Setup

## Requirements

- **Windows PowerShell**
- **Python 3.10+**
- Internet access for:
  - Alpha Vantage
  - Yahoo Finance
  - Hugging Face
- An **Alpha Vantage API key** for news collection
- Optional CUDA-capable PyTorch installation for faster FinBERT inference

The direct Python dependencies are listed in [`requirements.txt`](requirements.txt).

---

## 1. Create the Virtual Environment

Run the following commands from the repository root:

```powershell
py -3.10 -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

## 2. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

If the `venv` directory is deleted, simply recreate it using the commands above.

> Project data, saved models, figures, and generated outputs are stored outside the virtual environment and will not be deleted when the environment is recreated.

## 3. Verify the Environment

```powershell
python -c "import pandas, sklearn, xgboost, torch, transformers; print('Environment ready')"
```

Expected output:

```text
Environment ready
```

---

## 🚀 Optional GPU Acceleration

FinBERT sentiment scoring can be computationally expensive on CPU.

If a CUDA-capable GPU is available, install the appropriate PyTorch build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

CPU execution remains fully supported.

---

# 🔑 API Configuration

Set the Alpha Vantage API key for the **current PowerShell session**:

```powershell
$env:ALPHAVANTAGE_API_KEY = "YOUR_KEY_HERE"
```

For a fresh data collection, the API key must be available before running the pipeline.

If the required raw news data already exists locally, collection can be skipped.

---

# ▶️ Running the Pipeline

The pipeline runner executes stages in dependency order and stops if a stage fails.

## Complete Run

```powershell
python run_pipeline.py
```

This performs:

1. News coverage check
2. News collection
3. Price-data construction
4. Sentiment processing
5. Graph construction
6. Ablation modelling
7. Statistical analysis
8. Diagnostic generation

A valid `ALPHAVANTAGE_API_KEY` is required for a fresh collection.

---

## Run Using Existing Data

If the raw news data has already been downloaded:

```powershell
python run_pipeline.py --skip-collection
```

To additionally skip the optional coverage probe:

```powershell
python run_pipeline.py --skip-collection --skip-coverage
```

---

## Run Without Post-Modelling Analysis

To build the features and models without running post-modelling diagnostics:

```powershell
python run_pipeline.py --skip-analysis
```

---

# 🧩 Core Pipeline Stages

Individual stages can also be executed directly:

```powershell
python scripts/newscoverage.py

python scripts/collectnews.py

python scripts/build_price.py

python scripts/build_sentiment.py

python scripts/build_graph.py

python scripts/ablation.py data/config3_features.csv
```

### Stage Responsibilities

| Stage | Script | Function |
|---|---|---|
| **1** | `newscoverage.py` | Checks whether available news coverage is sufficient |
| **2** | `collectnews.py` | Downloads Alpha Vantage responses using resumable time slices |
| **3** | `build_price.py` | Downloads adjusted OHLCV data and creates the target plus 16 price features |
| **4** | `build_sentiment.py` | Deduplicates articles, scores them with FinBERT, and aggregates sentiment/news activity |
| **5** | `build_graph.py` | Builds historical news co-occurrence graphs, trains Node2Vec, and aligns daily embeddings |
| **6** | `ablation.py` | Tunes Logistic Regression and XGBoost on validation data and evaluates the held-out test period |
| **7** | Diagnostics | Performs statistical testing and generates research figures |

---

# 📰 News & NLP Processing

`collectnews.py` downloads Alpha Vantage financial-news responses in resumable time slices.

The pipeline preserves the original responses:

```text
data/raw/news/
```

FinBERT sentiment scores are cached in:

```text
data/finbert_cache.csv
```

This avoids repeatedly scoring the same articles during development or pipeline reruns.

The sentiment stage:

1. Loads raw financial-news articles
2. Deduplicates articles
3. Applies FinBERT
4. Generates sentiment scores
5. Aggregates sentiment by ticker and date
6. Incorporates time-decayed news activity
7. Produces the Config 2 feature table

---

# 🕸️ Knowledge Graph & Node2Vec

The graph stage constructs **past-news co-occurrence networks** from entities appearing together in financial news.

The workflow is:

```text
Financial News
      │
      ▼
Entity Extraction
      │
      ▼
Co-occurrence Relationships
      │
      ▼
Historical News Graph
      │
      ▼
Node2Vec
      │
      ▼
Node Embeddings
      │
      ▼
Daily Graph Features
```

Graph embeddings are cached in:

```text
data/graph_emb_cache.pkl
```

The resulting graph features are added to the sentiment-enhanced feature set to create **Config 3**.

---

# 📊 Feature Configurations

## Config 1 — Technical Baseline

```text
Price-derived features
        +
Next-day direction target
```

Purpose:

> Establish how well stock-price information alone predicts next-day direction.

---

## Config 2 — Financial News & Sentiment

```text
Config 1
   +
FinBERT sentiment
   +
News activity features
```

Purpose:

> Determine whether financial-news information provides predictive value beyond historical price information.

---

## Config 3 — Knowledge Graph Enhancement

```text
Config 2
   +
Node2Vec graph embeddings
```

Purpose:

> Determine whether relational information captured by the financial-news co-occurrence graph provides additional predictive information beyond prices and sentiment.

---

# 🤖 Modelling

Two classifiers are evaluated across the three configurations:

### Logistic Regression

Provides a relatively interpretable linear baseline for binary direction classification.

### XGBoost

Provides a nonlinear tree-based model capable of capturing feature interactions and nonlinear relationships.

This produces **six primary ablation models**:

| Configuration | Logistic Regression | XGBoost |
|---|:---:|:---:|
| Config 1 | ✓ | ✓ |
| Config 2 | ✓ | ✓ |
| Config 3 | ✓ | ✓ |

---

# 🔬 Experimental Design

The modelling process follows a strict chronological structure:

```text
2022 ─────────────── 2023
       TRAINING
           │
           ▼
2024 H1
VALIDATION
           │
           ▼
2024 H2
 FINAL TEST
```

### Important Controls

- Training data is used for model fitting.
- Scalers are fitted **only on training data**.
- Hyperparameters are selected using the validation period.
- The final test period remains unseen until final evaluation.
- The target uses only the following day's close.
- Price features use information available by date `t`.

All feature tables use:

```text
(ticker, date)
```

as their panel key.

---

# 💾 Generated Data & Outputs

### Main Feature Tables

| File | Contents |
|---|---|
| `data/config1_features.csv` | Price features + target |
| `data/config2_features.csv` | Price + sentiment/news features + target |
| `data/config3_features.csv` | Price + sentiment/news + graph features + target |

### Model Bundles

The six selected ablation models are stored under:

```text
models/
```

Each `.joblib` bundle contains:

- Trained model
- Training scaler
- Feature-column metadata
- Selected parameters
- Split dates

---

# 🔁 Loading a Saved Model

Example:

```python
import joblib

bundle = joblib.load(
    "models/config2_price_sentiment_logisticregression.joblib"
)

probabilities = bundle["model"].predict_proba(
    bundle["scaler"].transform(
        feature_frame[bundle["feature_columns"]]
    )
)[:, 1]
```

The feature-column metadata stored with the model ensures that inference uses the same feature ordering as training.

---

# 📈 Statistical Analysis & Diagnostics

After the ablation models have been generated, the following analyses can be executed.

```powershell
python scripts/delong.py data/config3_features.csv

python scripts/audit.py data/config3_features.csv

python scripts/descrip_cols.py

python scripts/check_sectors.py

python scripts/graph_viz.py

python scripts/diag_feature.py

python scripts/diag_boxplots.py

python scripts/diag_roccurves.py

python scripts/diag_perstock.py

python scripts/diag_mcnemar.py
```

### Available Diagnostics

| Script | Purpose |
|---|---|
| `delong.py` | ROC-AUC comparison/statistical testing |
| `audit.py` | Data and feature audit |
| `descrip_cols.py` | Descriptive feature analysis |
| `check_sectors.py` | Sector coverage checks |
| `graph_viz.py` | Knowledge-graph visualisation |
| `diag_feature.py` | Feature diagnostics |
| `diag_boxplots.py` | Distribution/boxplot diagnostics |
| `diag_roccurves.py` | ROC-curve analysis |
| `diag_perstock.py` | Per-stock performance analysis |
| `diag_mcnemar.py` | McNemar model comparison |

Additional exploratory/support utilities include:

```text
co_occurence_probe.py
inspect_raw.py
macro_entities.py
```

---

# 📦 Output Locations

| Location | Contents |
|---|---|
| `data/` | Raw data, processed data, feature tables and caches |
| `models/` | Saved model and scaler bundles |
| `outputs/` | Predictions, statistical tests and diagnostics |
| `figures/` | General research figures |

---

# ♻️ Reproducibility

The project is designed to be reproducible across machines.

### Path Independence

All scripts derive paths from the repository location rather than relying on hard-coded absolute paths.

### Cached Computation

Expensive processing is cached:

```text
FinBERT scores
       ↓
data/finbert_cache.csv

Node2Vec embeddings
       ↓
data/graph_emb_cache.pkl
```

This means interrupted or repeated experiments do not necessarily require recomputing previously processed data.

### Model Reproducibility

Saved model bundles contain the information required for later prediction, including:

- Model
- Scaler
- Feature columns
- Selected parameters
- Split dates

---

# ⚠️ Statistical Considerations

The dataset is a **panel of multiple stocks observed across the same dates**.

Therefore, observations from the same trading date cannot necessarily be treated as fully independent.

The diagnostic framework includes **date-clustered bootstrap intervals** to account for this dependence structure.

Statistical results should therefore be interpreted with the panel structure in mind.

---

# 🛠️ Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Activate the virtual environment and reinstall dependencies:

```powershell
.\venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

---

### `ALPHAVANTAGE_API_KEY is required`

Set the API key:

```powershell
$env:ALPHAVANTAGE_API_KEY = "YOUR_KEY_HERE"
```

Or, if the required raw news data already exists:

```powershell
python run_pipeline.py --skip-collection
```

---

### Pipeline rerun is slow

The pipeline reuses cached:

- FinBERT sentiment scores
- Node2Vec graph embeddings

Check:

```text
data/finbert_cache.csv
data/graph_emb_cache.pkl
```

---

### A pipeline stage fails

Run the failed stage independently from the repository root.

For example:

```powershell
python scripts/build_sentiment.py
```

This makes it easier to inspect the exact input data and error message.

---

# 🔍 Research Integrity & Scope

This pipeline is designed for **academic research and reproducible experimentation**.

The models evaluate predictive relationships for the selected companies and study period. Results should not be interpreted as evidence of a guaranteed trading strategy.

> **The outputs are not investment advice.**

---

# 📚 Research Summary

### Core Research Question

> **Does incorporating financial-news sentiment and knowledge-graph representations improve next-day stock-price direction prediction beyond traditional price-derived features?**

### Experimental Logic

```text
PRICE INFORMATION
       │
       ▼
   Config 1
       │
       │ + News & FinBERT
       ▼
   Config 2
       │
       │ + Knowledge Graph & Node2Vec
       ▼
   Config 3
       │
       ▼
 Compare Predictive Performance
       │
       ├── Logistic Regression
       └── XGBoost
       │
       ▼
 Statistical Testing
       │
       ▼
 Research Conclusions
```

The nested design allows the incremental contribution of **financial news sentiment** and **relational graph information** to be evaluated systematically rather than comparing unrelated model configurations.

---

## 🧾 Citation

If this repository is used in academic work, please cite the associated dissertation/research project.

---

<div align="center">

**Financial News × NLP × Knowledge Graphs × Machine Learning**

*Reproducible research pipeline for next-day stock direction prediction.*

</div>
