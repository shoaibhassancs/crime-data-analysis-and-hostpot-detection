import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
MODEL_READY_FILE = f"{PROJECT_PATH}/data/processed/model_ready.csv"
MODELS = f"{PROJECT_PATH}/models"
REPORTS = f"{PROJECT_PATH}/reports"
VISUALS = f"{PROJECT_PATH}/visuals"

for folder in (MODELS, REPORTS, VISUALS):
    os.makedirs(folder, exist_ok=True)

pd.set_option("display.width", 200)

# -----------------------------
# Load the Model-Ready Data
# -----------------------------
model_ready = pd.read_csv(MODEL_READY_FILE)

target = "SEVERITY"
categorical_cols = ["CRIME_TYPE", "TOWN", "SUBDIVISION", "SUBDIVISION_RISK_LEVEL"]
numeric_cols = ["LATITUDE", "LONGITUDE", "YEAR", "MONTH", "DAY_OF_WEEK", "IS_WEEKEND"]
feature_cols = categorical_cols + numeric_cols

X = model_ready[feature_cols]
y = model_ready[target]

# Same split as in 05_baseline_model.py, so every model is tested on the same 20,000 rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# -----------------------------
# Helper Functions
# -----------------------------
# Turn text into 0/1 columns and (optionally) scale the numbers
def make_preparation(scale_numbers):
    number_step = StandardScaler() if scale_numbers else "passthrough"
    return ColumnTransformer([
        ("text_columns", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("number_columns", number_step, numeric_cols),
    ])

# Score a model on the test data
def score_on_test(model):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")  # treats Low, Medium and High equally
    return predictions, round(accuracy, 3), round(macro_f1, 3)

# -----------------------------
# Baseline (for comparison)
# -----------------------------
baseline_model = Pipeline([
    ("prepare", make_preparation(scale_numbers=True)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
baseline_model.fit(X_train, y_train)

_, baseline_accuracy, baseline_f1 = score_on_test(baseline_model)
print(f"\nBaseline logistic regression: accuracy {baseline_accuracy} | macro F1 {baseline_f1}")

# -----------------------------
# Tune Logistic Regression
# -----------------------------
# Cross-validation: the training data is split into 3 parts. The model is trained on 2 parts
# and scored on the 3rd, three times. The test data is never touched while tuning
print("\nTuning logistic regression...")
start = time.time()

logistic_search = GridSearchCV(
    Pipeline([
        ("prepare", make_preparation(scale_numbers=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},  # C = how strongly the model is held back from overfitting
    cv=3, scoring="f1_macro", n_jobs=1,
)
logistic_search.fit(X_train, y_train)

logistic_index = logistic_search.best_index_
logistic_cv = logistic_search.best_score_
logistic_cv_std = logistic_search.cv_results_["std_test_score"][logistic_index]

print(f"Done in {time.time() - start:.0f}s. Best settings: {logistic_search.best_params_}")
print(f"Cross-validated macro F1: {logistic_cv:.3f} (+/- {logistic_cv_std:.3f})")

# -----------------------------
# Tune Random Forest
# -----------------------------
print("\nTuning random forest (this is the slow step)...")
start = time.time()

forest_search = RandomizedSearchCV(
    Pipeline([
        ("prepare", make_preparation(scale_numbers=False)),
        ("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)),
    ]),
    param_distributions={
        "model__n_estimators": [100, 200],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__max_depth": [None, 20, 40],
        "model__max_features": ["sqrt", 0.3],
    },
    n_iter=6, cv=3, scoring="f1_macro", random_state=42, n_jobs=1,
)
forest_search.fit(X_train, y_train)

forest_index = forest_search.best_index_
forest_cv = forest_search.best_score_
forest_cv_std = forest_search.cv_results_["std_test_score"][forest_index]

print(f"Done in {time.time() - start:.0f}s. Best settings: {forest_search.best_params_}")
print(f"Cross-validated macro F1: {forest_cv:.3f} (+/- {forest_cv_std:.3f})")

# -----------------------------
# Choose the Final Model
# -----------------------------
# The choice uses cross-validation only, never the test data.
# The simpler model wins unless the forest is clearly better (by more than 0.005)
margin = 0.005

if forest_cv > logistic_cv + margin:
    final_name = "Random forest (tuned)"
    final_search = forest_search
else:
    final_name = "Logistic regression (tuned)"
    final_search = logistic_search

final_model = final_search.best_estimator_

print(f"\nFinal model: {final_name}")

# -----------------------------
# Test the Models on Unseen Data
# -----------------------------
dummy_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

rows = []
candidates = [
    ("Always predict High", dummy_model, None, None),
    ("Logistic regression (baseline)", baseline_model, None, None),
    ("Logistic regression (tuned)", logistic_search.best_estimator_, logistic_cv, logistic_cv_std),
    ("Random forest (tuned)", forest_search.best_estimator_, forest_cv, forest_cv_std),
]

for name, model, cv_score, cv_std in candidates:
    _, accuracy, macro_f1 = score_on_test(model)
    rows.append({
        "Model": name,
        "CV macro F1": round(cv_score, 3) if cv_score is not None else "",
        "Test accuracy": accuracy,
        "Test macro F1": macro_f1,
        "Chosen": "yes" if name == final_name else "",
    })

results = pd.DataFrame(rows)
print("\nResults (test = 20,000 rows the models have never seen):")
print(results.to_string(index=False))

# -----------------------------
# Testing Report for the Final Model
# -----------------------------
final_predictions, final_accuracy, final_f1 = score_on_test(final_model)

labels = ["Low", "Medium", "High"]
report = classification_report(y_test, final_predictions, labels=labels, digits=3)
matrix = confusion_matrix(y_test, final_predictions, labels=labels)

print(f"\nClassification report ({final_name}):")
print(report)

print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(matrix, index=labels, columns=labels))

# -----------------------------
# Where Does the Model Fall Short?
# -----------------------------
# Errors that jump two levels (Low <-> High) are worse than errors between neighbours
total_errors = int(matrix.sum() - matrix.trace())
two_level_errors = int(matrix[0, 2] + matrix[2, 0])

print(f"\nTotal errors: {total_errors} of {len(y_test)} ({total_errors / len(y_test):.1%})")
print(f"Errors that jump from Low to High or High to Low: {two_level_errors}")

# Accuracy for different parts of the test data
test_results = X_test.copy()
test_results["actual"] = y_test
test_results["predicted"] = final_predictions
test_results["correct"] = (test_results["actual"] == test_results["predicted"]).astype(int)

risk_order = ["Low", "Medium", "Medium-High", "High"]
accuracy_by_risk = (test_results.groupby("SUBDIVISION_RISK_LEVEL")["correct"]
                    .agg(rows="size", accuracy="mean").reindex(risk_order).round(3))
print("\nAccuracy by area risk level:")
print(accuracy_by_risk)

accuracy_by_crime = (test_results.groupby("CRIME_TYPE")["correct"]
                     .agg(rows="size", accuracy="mean").sort_values("accuracy").round(3))
print("\n8 crime types the model gets wrong most often:")
print(accuracy_by_crime.head(8))

# -----------------------------
# Confusion Matrix Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
    ax=ax, cmap="Blues", values_format=",d", colorbar=False
)
ax.set_title(f"{final_name}\npredicted vs actual severity (test data)")
ax.set_xlabel("Predicted severity")
ax.set_ylabel("Actual severity")
fig.tight_layout()
fig.savefig(f"{VISUALS}/06_01_tuned_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Save the Results and the Model
# -----------------------------
results.to_csv(f"{REPORTS}/06_01_model_results.csv", index=False)

with open(f"{REPORTS}/06_02_tuned_classification_report.txt", "w") as file:
    file.write(f"{final_name}\n")
    file.write(f"Best settings: {final_search.best_params_}\n\n")
    file.write(report)

accuracy_by_risk.to_csv(f"{REPORTS}/06_03_accuracy_by_risk_level.csv")
accuracy_by_crime.to_csv(f"{REPORTS}/06_04_accuracy_by_crime_type.csv")

joblib.dump(final_model, f"{MODELS}/tuned_model.joblib")

print("\nSaved: reports/06_01_model_results.csv")
print("Saved: reports/06_02_tuned_classification_report.txt")
print("Saved: reports/06_03_accuracy_by_risk_level.csv")
print("Saved: reports/06_04_accuracy_by_crime_type.csv")
print("Saved: visuals/06_01_tuned_confusion_matrix.png")
print("Saved: models/tuned_model.joblib")