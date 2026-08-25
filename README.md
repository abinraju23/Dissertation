# Dissertation Pipeline

This repository contains the data-collection, feature-engineering, modelling,
and diagnostic scripts for the dissertation study:

> Leveraging Financial News Knowledge Graphs & NLP for Short-Term Stock Price Direction Classification

## Requirements

- Windows PowerShell (the commands below use PowerShell syntax)
- Python 3.10 or later
- Internet access for Alpha Vantage, Yahoo Finance, and Hugging Face
- An Alpha Vantage API key for news collection
- Optional CUDA-capable PyTorch installation for faster FinBERT scoring

The direct Python dependencies are listed in [requirements.txt](requirements.txt).
Run all commands from the repository root.

## Environment Setup

Create and activate the project virtual environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

For GPU acceleration, install the appropriate CUDA build of PyTorch using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing the remaining requirements. CPU execution is supported but FinBERT
sentiment scoring can be slow.

Set the Alpha Vantage key for the current PowerShell session:

```powershell
$env:ALPHAVANTAGE_API_KEY = "YOUR_KEY_HERE"
```

## Pipeline

Run the complete pipeline from the repository root with:

```powershell
python run_pipeline.py
```

The runner executes data collection, feature engineering, ablation modelling,
statistical tests, and diagnostics in dependency order. It stops if a stage
fails. To use existing raw news, skip API collection; to omit the optional
coverage probe or the diagnostics, use the corresponding switches:

```powershell
python run_pipeline.py --skip-coverage
python run_pipeline.py --skip-collection
python run_pipeline.py --skip-analysis
```

The equivalent core stages are:

```powershell
python scripts/newscoverage.py
python scripts/collectnews.py
python scripts/build_price.py
python scripts/build_sentiment.py
python scripts/build_graph.py
python scripts/ablation.py data/config3_features.csv
```

The first command is an optional news-coverage probe. `collectnews.py` can be
resumed, and FinBERT scores are cached in `data/finbert_cache.csv`.

The main generated inputs are:

- `data/raw/news/`: downloaded Alpha Vantage news
- `data/prices/`: per-stock price data
- `data/config1_features.csv`: price features and target
- `data/config2_features.csv`: price plus sentiment features
- `data/config3_features.csv`: price, sentiment, and graph features

## Analysis and Diagnostics

After the ablation run, use the following scripts as needed:

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

Support and exploratory utilities include `co_occurence_probe.py`,
`inspect_raw.py`, and `macro_entities.py`.

Modelling tables, predictions, statistical tests, and diagnostic figures are
written under `outputs/`. The six selected ablation models, together with their
training scalers and feature-column metadata, are saved as `.joblib` bundles
under `models/`. General figures are stored under `figures/`.

Each saved model can be loaded with:

```python
import joblib

bundle = joblib.load("models/config2_price_sentiment_logisticregression.joblib")
probabilities = bundle["model"].predict_proba(
	bundle["scaler"].transform(feature_frame[bundle["feature_columns"]])
)[:, 1]
```

## Reproducibility Notes

The ablation script uses chronological train, validation, and test periods.
Scalers are fitted on training data only, hyperparameters are selected on the
validation period, and the test period is reserved for final scoring.

All pipeline scripts now derive paths from the repository location, so the
project can be moved to another machine without editing source paths. Existing
generated CSV files can still be used directly by the analysis scripts.
