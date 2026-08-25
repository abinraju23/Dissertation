# Dissertation Pipeline

## Leveraging financial news, NLP, and knowledge graphs for stock direction prediction

This project investigates whether financial news sentiment and a news
co-occurrence knowledge graph improve short-term stock price direction
classification. It builds a reproducible panel of 30 US-listed companies,
creates three nested feature configurations, trains two classifiers, and
evaluates them using a chronological holdout design.

The three configurations are:

| Configuration | Feature block | Purpose |
| --- | --- | --- |
| Config 1 | Price-derived features | Technical baseline |
| Config 2 | Config 1 plus FinBERT and news features | Test the contribution of sentiment |
| Config 3 | Config 2 plus Node2Vec graph features | Test the contribution of relational structure |

The target is the next trading day's price direction. For a row at day `t`, the
label is `1` when the next close is higher than the current close, and `0`
otherwise. Training uses 2022-2023, validation uses 2024 H1, and final testing
uses 2024 H2.

## Project structure

```text
data/             Raw news, prices, feature tables, and generated figures
figures/          General project figures
models/           Saved model and scaler bundles
outputs/          Results, predictions, diagnostics, and statistical tables
scripts/          Collection, feature engineering, modelling, and diagnostics
run_pipeline.py   Ordered pipeline runner
```

The original Alpha Vantage responses are retained under `data/raw/news/` so
later feature-building stages can be inspected or rerun.

## Requirements

- Windows PowerShell (the commands below use PowerShell syntax)
- Python 3.10 or later
- Internet access for Alpha Vantage, Yahoo Finance, and Hugging Face
- An Alpha Vantage API key for news collection
- Optional CUDA-capable PyTorch installation for faster FinBERT scoring

The direct Python dependencies are listed in [requirements.txt](requirements.txt).
Run all commands from the repository root. The full pipeline requires PyTorch
and Transformers for FinBERT, XGBoost for the ablation models, and internet
access when downloading prices, news, or the FinBERT model.

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

If the `venv` folder is deleted, recreate it from the project root using the
same commands above. The project data, saved models, and generated outputs are
stored outside the virtual environment and are not deleted with it.

To verify that the recreated environment is ready:

```powershell
python -c "import pandas, sklearn, xgboost, torch, transformers; print('Environment ready')"
```

For optional GPU acceleration, install the appropriate CUDA build of PyTorch using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing the remaining requirements. CPU execution is supported but FinBERT
sentiment scoring can be slow.

Set the Alpha Vantage key for the current PowerShell session:

```powershell
$env:ALPHAVANTAGE_API_KEY = "YOUR_KEY_HERE"
```

To run the pipeline with existing downloaded data, skip news collection:

```powershell
python run_pipeline.py --skip-collection
```

To also skip the optional coverage probe:

```powershell
python run_pipeline.py --skip-collection --skip-coverage
```

For a fresh collection, set `ALPHAVANTAGE_API_KEY` in the current PowerShell
session before running `python run_pipeline.py`.

## Running the pipeline

The runner executes each stage in dependency order and stops if a stage fails.

### Complete run

```powershell
python run_pipeline.py
```

This includes data collection, feature engineering, ablation modelling,
statistical tests, and diagnostics. It requires `ALPHAVANTAGE_API_KEY`.

### Run with existing data

When raw news is already available locally, skip the collection stage:

```powershell
python run_pipeline.py --skip-collection
```

To also skip the optional coverage probe:

```powershell
python run_pipeline.py --skip-collection --skip-coverage
```

To build features and models without the post-modelling diagnostics:

```powershell
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

The coverage probe is optional. `collectnews.py` can be resumed, FinBERT scores
are cached in `data/finbert_cache.csv`, and graph embeddings are cached in
`data/graph_emb_cache.pkl`.

The main generated inputs are:

- `data/raw/news/`: downloaded Alpha Vantage news
- `data/prices/`: per-stock price data
- `data/config1_features.csv`: price features and target
- `data/config2_features.csv`: price plus sentiment features
- `data/config3_features.csv`: price, sentiment, and graph features

## Analysis and diagnostics

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

## Processing stages

1. `newscoverage.py` checks whether the available news coverage is sufficient.
2. `collectnews.py` downloads Alpha Vantage responses in resumable time slices.
3. `build_price.py` downloads adjusted OHLCV data and creates the shared target
	and 16 price features.
4. `build_sentiment.py` deduplicates articles, scores them with FinBERT, and
	aggregates time-decayed sentiment and news activity by ticker and date.
5. `build_graph.py` creates past-news co-occurrence graphs, trains Node2Vec,
	aligns daily embeddings, and adds graph features.
6. `ablation.py` tunes Logistic Regression and XGBoost on validation data and
	scores the held-out test period.
7. `delong.py` and the diagnostic scripts analyse stored predictions and write
	statistical tables and figures.

All feature tables use `(ticker, date)` as their panel key. Price features use
information available by date `t`; only the target uses the following close.

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

## Reproducibility notes

The ablation script uses chronological train, validation, and test periods.
Scalers are fitted on training data only, hyperparameters are selected on the
validation period, and the test period is reserved for final scoring.

All pipeline scripts derive paths from the repository location, so the project
can be moved to another machine without editing source paths. The saved model
bundles include the scaler, feature-column list, selected parameters, and split
dates needed for later prediction.

The final test period is not used for hyperparameter selection. Statistical
results should be interpreted carefully because the panel contains multiple
stocks on each date, so observations from the same date are not fully
independent. The diagnostics include date-clustered bootstrap intervals for
this reason.

The models answer a research question for the selected stocks and period. They
are not investment advice or a complete trading strategy.

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'`: activate the virtual
	environment and install the dependencies with
	`python -m pip install -r requirements.txt`.
- `ALPHAVANTAGE_API_KEY is required`: set the key in the current PowerShell
	session, or use `--skip-collection` when raw news is already available.
- A rerun is slow: FinBERT scores and graph embeddings are reused from their
	cache files when available.
- A stage fails: run that stage directly from the repository root to inspect
	its input table and error message.
