# Predicting Crime Severity: Model Report

## Goal
Predict whether a recorded incident is **Low**, **Medium** or **High** severity from where it happened, what type of crime it was, and when. The data is the Kaggle Karachi crime dataset (100,000 incidents, Jun 2020 - May 2025). It is **synthetic**, so these results show the method, not real-world crime in Karachi.

## Data preparation
- Cleaned data from `02_data_cleaning.py` (no missing values, no duplicates, labels standardised, dates converted).
- Target: `SEVERITY`.
- **Not used as inputs:**
  - `SEVERITY_SCORE`: it is the target written as numbers (1-4 Low, 5-7 Medium, 8-10 High), so using it would be cheating.
  - `INCIDENT_ID`.
  - `RISK_ZONE`: it carries the same information as `SUBDIVISION_RISK_LEVEL`.
  - `RANK`: its meaning is not documented.
  - Text copies of the date columns.
- **Inputs used:** crime type, town, subdivision, subdivision risk level, latitude, longitude, year, month, day of week, weekend flag.

## How the models were tested
- 80% of rows for training (80,000) and 20% for testing (20,000), split once with a fixed seed and the same Low/Medium/High mix in both parts.
- All tuning used cross-validation on the training rows only (3 folds). The 20,000 test rows were used only for the final scores.
- Every model in this report was scored on the same 20,000 test rows.
- **Macro F1** is reported next to accuracy. It scores Low, Medium and High equally, so a model cannot look good by only predicting the most common class (High, 55% of rows).

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Always predict High (no learning) | 0.551 | 0.237 |
| Rule: crime type only | 0.660 | 0.605 |
| Rule: crime type + subdivision | 0.899 | 0.881 |
| **Logistic regression (chosen)** | **0.894** | **0.876** |
| Decision tree | 0.876 | 0.864 |
| Random forest | 0.890 | 0.871 |
| Gradient boosting | 0.894 | 0.875 |

## What the results mean
1. **Severity is predictable, up to a ceiling of about 0.89 accuracy.** The best score is about 0.88-0.90 for every approach that uses both crime type and location.
2. **More complex models did not help.** Tuned trees, a random forest and gradient boosting all land at or below plain logistic regression. The simplest model that reaches the ceiling was chosen: it is faster, easier to explain and cannot overfit as easily.
3. **Even a lookup table reaches the same ceiling.** The best "most common severity per crime type and subdivision" rule scores 0.899 / 0.881, which is within noise of the models. The model adds value by handling combinations not seen in training, giving probabilities and explaining its decisions.
4. **Errors are only between neighbouring levels.** About 10.6% of test incidents are misclassified. Low is never confused with High (see `06_02_tuned_classification_report.txt` and the confusion matrix `06_01_tuned_confusion_matrix.png`).

## What drives the predictions
Shuffling a group of columns and measuring the drop in macro F1 (bigger drop = more reliance):

| Column group | Drop in macro F1 |
|---|---|
| Location (town, subdivision, risk level, coordinates) | 0.400 |
| Crime type | 0.385 |
| Time (year, month, weekday, weekend) | 0.0003 (no effect) |

- Severity depends almost equally on **where** and **what**.
- **When** an incident happened has no effect on its severity.
- Crime types pushing severity toward High: Murder, Gang Violence, Homicide, Kidnapping, Political Violence, Assault.
- Crime types pushing toward Low: Industrial Hazards, Drug Possession, Gang Activity, Theft, Snatching, Land Grabbing.
- Treat the crime-type values as a **ranking**, not exact amounts (`07_03_crime_type_push.csv`).

## Limits
- The data is synthetic, and severity looks like a score built from crime type plus area risk. Real data would be noisier and the ceiling would likely be lower.
- The model labels the severity of incidents that were already recorded. It does not forecast when or where future crimes will happen. Time has no signal in this data, so no forecast would be reliable.
- Coordinates are neighbourhood-level (161 distinct points), so they add little beyond subdivision.

## Files
| File | What it is |
|---|---|
| `notebooks/05_baseline_model.py` | Baseline (Basic tier) |
| `notebooks/06_tuned_model_and_testing.py` | Tuned model and honest testing (Standard tier) |
| `notebooks/07_model_comparison_and_explanation.py` | Model comparison and explanation (Premium tier) |
| `models/tuned_model.joblib` | The final model, ready to load and use |
| `reports/05_*`, `06_*`, `07_*` | Result tables and reports |
| `visuals/05_*`, `06_*`, `07_*` | Charts |