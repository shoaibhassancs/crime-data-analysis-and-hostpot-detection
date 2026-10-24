# Karachi Crime Data: Cleaning, Analysis, Dashboard and Severity Model

An end-to-end data project in Python: a raw crime dataset is audited and cleaned, explored with charts, turned into an interactive dashboard, and used to build and test a machine learning model that predicts incident severity (Low / Medium / High).

> **About the data:** the Kaggle dataset used here is **synthetic** (its SOURCE column says "Synthetic"). The results show the method and workflow, not real crime levels in Karachi.

## Headline results
- 100,000 incidents, Jun 2020 - May 2025, 18 towns, 194 subdivisions, 27 crime types
- Cleaning: no missing values or duplicates found; 5,607 inconsistent labels standardised; 562 rows with wrong coordinates flagged (kept for counts, left off maps); redundant columns removed
- Severity can be predicted with about **89% accuracy** (macro F1 0.88) from crime type and location
- Time of year, month and weekday have **no effect** on severity
- A simple logistic regression matches tuned random forests and gradient boosting. More complex models added nothing

## Project structure
```
data/raw/          the original Kaggle CSV (not included, see below)
data/processed/    cleaned data and model-ready data
notebooks/         numbered scripts, run in order
reports/           cleaning log, summary tables, model results, written reports
visuals/           charts (named after the script that made them)
dashboard/         interactive HTML dashboard
models/            saved models
```

## Scripts
| Script | What it does |
|---|---|
| `01_data_overview.py` | First look: shape, types, missing values, duplicates, constant columns |
| `02_data_cleaning.py` | Cleaning with a step-by-step log (`reports/cleaning_log.csv`) |
| `03_charts.py` | 9 charts (`visuals/03_*`) |
| `04_exploration_findings.py` | Summary tables and unusual days (`reports/04_*`) |
| `05_baseline_model.py` | Data prep and a baseline model |
| `06_tuned_model_and_testing.py` | Cross-validated tuning and honest testing on unseen data |
| `07_model_comparison_and_explanation.py` | Seven approaches compared, plus what drives the predictions |
| `08_dashboard.py` | Builds the interactive dashboard (`dashboard/08_dashboard.html`) |

## How to run
1. Download the dataset from Kaggle ([add the Kaggle link here]) and save it as `data/raw/karachi_crime_2020_2025.csv`.
2. Create an environment and install the packages (Python 3.13 was used):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Open each script and check that `PROJECT_PATH` at the top points to your project folder.
4. Run the scripts in order:
```bash
python notebooks/01_data_overview.py
python notebooks/02_data_cleaning.py
# ...and so on up to 08_dashboard.py
```
5. Open `dashboard/08_dashboard.html` in a browser (it needs internet to load the chart library).

## Written reports
- `reports/findings_summary.md`: one-page summary of the data findings
- `reports/model_report.md`: how the model was built, tested and what it can and cannot do

## Limits
- The data is synthetic, so real-world conclusions should not be drawn from it.
- The model labels the severity of incidents already recorded. It does not forecast future crime.

## Tools
Python, pandas, scikit-learn, matplotlib, Chart.js