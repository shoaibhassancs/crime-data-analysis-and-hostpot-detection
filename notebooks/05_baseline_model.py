import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"
MODEL_READY_FILE = f"{PROJECT_PATH}/data/processed/model_ready.csv"
MODELS = f"{PROJECT_PATH}/models"
REPORTS = f"{PROJECT_PATH}/reports"
VISUALS = f"{PROJECT_PATH}/visuals"

for folder in (MODELS, REPORTS, VISUALS):
    os.makedirs(folder, exist_ok=True)

pd.set_option("display.width", 200)

# -----------------------------
# Load the Cleaned Data
# -----------------------------
df = pd.read_csv(CLEAN_FILE)

print("Rows and columns:", df.shape)

# -----------------------------
# Define the Target and Features
# -----------------------------
# Goal: predict how severe an incident is (Low / Medium / High)
target = "SEVERITY"

# Text columns that the model will turn into 0/1 columns (one-hot encoding)
categorical_cols = ["CRIME_TYPE", "TOWN", "SUBDIVISION", "SUBDIVISION_RISK_LEVEL"]

# Number columns (scaled later so no column dominates)
numeric_cols = ["LATITUDE", "LONGITUDE", "YEAR", "MONTH", "DAY_OF_WEEK", "IS_WEEKEND"]

# Columns left out on purpose:
# - SEVERITY_SCORE is SEVERITY in numbers, so using it would be cheating
# - RISK_ZONE repeats SUBDIVISION_RISK_LEVEL
# - INCIDENT_ID, DATE, YEAR_MONTH, DAY_NAME, PERIOD are IDs or copies of other columns
# - RANK and the priority ranks have no clear meaning
# - IS_SUSPECT_COORD is only a data-quality flag
feature_cols = categorical_cols + numeric_cols

# -----------------------------
# Create the Model-Ready Data
# -----------------------------
model_ready = df[["INCIDENT_ID"] + feature_cols + [target]]

print("\nMissing values in model-ready data:", int(model_ready.isnull().sum().sum()))
print("Target classes (%):")
print((model_ready[target].value_counts(normalize=True) * 100).round(1))

model_ready.to_csv(MODEL_READY_FILE, index=False)

# -----------------------------
# Train / Test Split
# -----------------------------
X = model_ready[feature_cols]
y = model_ready[target]

# 80% to learn from, 20% kept aside to test on data the model has never seen.
# stratify keeps the same share of Low / Medium / High in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining rows:", len(X_train))
print("Test rows:", len(X_test))

# -----------------------------
# Helper Function
# -----------------------------
results = []

# Score a model on the test data and store the result
def evaluate(name, model):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")  # treats Low, Medium and High equally

    results.append({"Model": name, "Accuracy": round(accuracy, 3), "Macro F1": round(macro_f1, 3)})
    print(f"{name:<32} accuracy {accuracy:.3f} | macro F1 {macro_f1:.3f}")
    return predictions

# -----------------------------
# Baseline 1: Always Predict the Most Common Class
# -----------------------------
# The simplest possible "model". Any real model has to beat this
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

print()
evaluate("Always predict High", dummy_model)

# -----------------------------
# Baseline 2: Logistic Regression
# -----------------------------
# The pipeline does three things in one object: turn text into 0/1 columns,
# scale the numbers, then fit the model. The test data goes through
# exactly the same steps as the training data
preparation = ColumnTransformer([
    ("text_columns", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("number_columns", StandardScaler(), numeric_cols),
])

baseline_model = Pipeline([
    ("prepare", preparation),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

baseline_model.fit(X_train, y_train)

baseline_predictions = evaluate("Logistic regression (baseline)", baseline_model)

# -----------------------------
# Baseline Test Report
# -----------------------------
labels = ["Low", "Medium", "High"]

report = classification_report(y_test, baseline_predictions, labels=labels, digits=3)
print("\nClassification report (logistic regression):")
print(report)

matrix = confusion_matrix(y_test, baseline_predictions, labels=labels)
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(matrix, index=labels, columns=labels))

# -----------------------------
# Confusion Matrix Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
    ax=ax, cmap="Blues", values_format=",d", colorbar=False
)
ax.set_title("Baseline model: predicted vs actual severity")
ax.set_xlabel("Predicted severity")
ax.set_ylabel("Actual severity")
fig.tight_layout()
fig.savefig(f"{VISUALS}/05_01_baseline_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Save the Results and the Model
# -----------------------------
pd.DataFrame(results).to_csv(f"{REPORTS}/05_01_baseline_results.csv", index=False)

with open(f"{REPORTS}/05_02_baseline_classification_report.txt", "w") as file:
    file.write(report)

joblib.dump(baseline_model, f"{MODELS}/baseline_logistic_regression.joblib")

print("\nSaved:", MODEL_READY_FILE)
print("Saved: reports/05_01_baseline_results.csv")
print("Saved: reports/05_02_baseline_classification_report.txt")
print("Saved: visuals/05_01_baseline_confusion_matrix.png")
print("Saved: models/baseline_logistic_regression.joblib")