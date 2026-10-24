import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

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
pd.set_option("display.max_rows", 100)

# -----------------------------
# Chart Style (same as 03_charts.py)
# -----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
DARK_BLUE = "#104281"
NOTE = "Synthetic data from Kaggle (Karachi crime), Jun 2020 - May 2025"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10,
})

# Function to finish and save a chart
def save_chart(fig, filename):
    fig.text(0.01, 0.01, NOTE, fontsize=8, color=INK_SOFT, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{VISUALS}/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", filename)

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

# Same split as in 05 and 06, so every model is tested on the same 20,000 rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# -----------------------------
# Helper Functions
# -----------------------------
# Turn text into 0/1 columns and (optionally) scale the numbers
def make_preparation(scale_numbers, dense=False):
    number_step = StandardScaler() if scale_numbers else "passthrough"
    return ColumnTransformer([
        ("text_columns", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), categorical_cols),
        ("number_columns", number_step, numeric_cols),
    ], sparse_threshold=0)

# Simple rule: for each combination of columns, predict the most common severity seen in training.
# Combinations never seen in training get "High" (the most common class overall)
def lookup_predictions(columns):
    train_table = X_train[columns].copy()
    train_table["SEVERITY"] = y_train.values
    most_common = train_table.groupby(columns)["SEVERITY"].agg(lambda s: s.mode().iloc[0])

    if len(columns) > 1:
        keys = pd.MultiIndex.from_frame(X_test[columns])
    else:
        keys = pd.Index(X_test[columns[0]])
    return most_common.reindex(keys).fillna("High").values

results = []

# Score predictions on the test data and store the result
def add_result(name, predictions, description):
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")  # treats Low, Medium and High equally
    results.append({"Model": name, "Accuracy": round(accuracy, 3),
                    "Macro F1": round(macro_f1, 3), "What it does": description})
    print(f"{name:<40} accuracy {accuracy:.3f} | macro F1 {macro_f1:.3f}", flush=True)

# -----------------------------
# Compare Several Models
# -----------------------------
print("\nTraining and testing models...\n")

# 1. Always predict the most common class
dummy_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
add_result("Always predict High", dummy_model.predict(X_test), "No learning at all")

# 2. Simple rules (no machine learning)
add_result("Rule: crime type only", lookup_predictions(["CRIME_TYPE"]),
           "Most common severity for each crime type")
add_result("Rule: crime type + subdivision", lookup_predictions(["CRIME_TYPE", "SUBDIVISION"]),
           "Most common severity for each crime type in each subdivision")

# 3. Logistic regression (settings from 06_tuned_model_and_testing.py)
logistic_model = Pipeline([
    ("prepare", make_preparation(scale_numbers=True)),
    ("model", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")),
]).fit(X_train, y_train)
add_result("Logistic regression", logistic_model.predict(X_test), "Adds up the effect of each column")

# 4. Decision tree (a small tree you can read)
tree_model = Pipeline([
    ("prepare", make_preparation(scale_numbers=False)),
    ("model", DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                     class_weight="balanced", random_state=42)),
]).fit(X_train, y_train)
add_result("Decision tree", tree_model.predict(X_test), "One tree of yes/no questions")

# 5. Random forest (settings from 06_tuned_model_and_testing.py)
forest_model = Pipeline([
    ("prepare", make_preparation(scale_numbers=False)),
    ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.3,
                                     max_depth=20, class_weight="balanced",
                                     n_jobs=-1, random_state=42)),
]).fit(X_train, y_train)
add_result("Random forest", forest_model.predict(X_test), "Many trees that vote")

# 6. Gradient boosting (each new tree fixes the mistakes of the previous ones)
start = time.time()
boosting_model = Pipeline([
    ("prepare", make_preparation(scale_numbers=False, dense=True)),
    ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
]).fit(X_train, y_train)
add_result("Gradient boosting", boosting_model.predict(X_test), "Trees that correct each other")
print(f"(gradient boosting took {time.time() - start:.0f}s)")

comparison = pd.DataFrame(results)

print("\nModel comparison (test = 20,000 rows the models have never seen):")
print(comparison.to_string(index=False))

comparison.to_csv(f"{REPORTS}/07_01_model_comparison.csv", index=False)

# -----------------------------
# Model Comparison Chart
# -----------------------------
ordered = comparison.sort_values("Macro F1")

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(ordered["Model"], ordered["Macro F1"], height=0.65, color=BLUE)

for i, value in enumerate(ordered["Macro F1"].values):
    ax.text(value, i, f"  {value:.3f}", va="center", fontsize=9)

ax.set_xlim(0, 1.05)
ax.grid(axis="x")
ax.set_title("Model comparison")
ax.set_xlabel("Macro F1 on unseen test data (higher is better, 1 is perfect)")

save_chart(fig, "07_01_model_comparison.png")

# -----------------------------
# What Drives the Predictions?
# -----------------------------
# Load the final model chosen in step 06
final_model = joblib.load(f"{MODELS}/tuned_model.joblib")

# Shuffle a group of columns together and see how much the score drops.
# A big drop means the model relies on that group. Related columns
# (for example town, subdivision and coordinates) are shuffled as one group
def group_importance(columns, repeats=5):
    base_score = f1_score(y_test, final_model.predict(X_test), average="macro")
    drops = []

    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X_test))

        shuffled = X_test.copy()
        for col in columns:
            shuffled[col] = X_test[col].values[order]

        score = f1_score(y_test, final_model.predict(shuffled), average="macro")
        drops.append(base_score - score)

    return np.mean(drops), np.std(drops)

groups = {
    "Crime type": ["CRIME_TYPE"],
    "Location (town, subdivision, risk level, coordinates)":
        ["TOWN", "SUBDIVISION", "SUBDIVISION_RISK_LEVEL", "LATITUDE", "LONGITUDE"],
    "Time (year, month, weekday, weekend)": ["YEAR", "MONTH", "DAY_OF_WEEK", "IS_WEEKEND"],
}

importance_rows = []
for group_name, columns in groups.items():
    mean_drop, std_drop = group_importance(columns)
    importance_rows.append({"Group": group_name, "Drop in macro F1": round(mean_drop, 4),
                            "Std": round(std_drop, 4)})

importance = pd.DataFrame(importance_rows).sort_values("Drop in macro F1", ascending=False)

print("\nHow much the score drops when a group of columns is shuffled:")
print(importance.to_string(index=False))

importance.to_csv(f"{REPORTS}/07_02_feature_group_importance.csv", index=False)

plot_data = importance.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(plot_data["Group"], plot_data["Drop in macro F1"], xerr=plot_data["Std"],
        height=0.55, color=DARK_BLUE, error_kw={"ecolor": INK_SOFT, "capsize": 3})

for i, (value, spread) in enumerate(zip(plot_data["Drop in macro F1"].values, plot_data["Std"].values)):
    ax.text(max(value, 0) + spread + 0.008, i, f"{abs(max(value, 0)):.3f}", va="center", fontsize=9)

ax.set_xlim(-0.005, plot_data["Drop in macro F1"].max() * 1.15)  # leave room for the labels
ax.axvline(0, color=INK_SOFT, linewidth=0.8)
ax.grid(axis="x")
ax.set_title("What drives severity predictions")
ax.set_xlabel("Drop in macro F1 when the columns are shuffled (bigger = the model relies on them more)")

save_chart(fig, "07_02_what_drives_severity.png")

# -----------------------------
# Which Crime Types Push Severity Up or Down?
# -----------------------------
# Only possible when the final model is logistic regression, because it has coefficients
final_step = final_model.named_steps["model"]

if hasattr(final_step, "coef_"):
    feature_names = final_model.named_steps["prepare"].get_feature_names_out()
    classes = list(final_step.classes_)

    # Positive = pushes toward High severity, negative = pushes toward Low severity
    push = final_step.coef_[classes.index("High")] - final_step.coef_[classes.index("Low")]

    prefix = "text_columns__CRIME_TYPE_"
    crime_push = pd.DataFrame({"feature": feature_names, "push": push})
    crime_push = crime_push[crime_push["feature"].str.startswith(prefix)].copy()
    crime_push["Crime type"] = crime_push["feature"].str.replace(prefix, "", regex=False)
    crime_push["Push toward High severity"] = crime_push["push"].round(2)
    crime_push = crime_push.sort_values("Push toward High severity", ascending=False)
    crime_push = crime_push[["Crime type", "Push toward High severity"]]

    print("\nCrime types that push severity up (top 8) and down (bottom 8):")
    print(pd.concat([crime_push.head(8), crime_push.tail(8)]).to_string(index=False))

    crime_push.to_csv(f"{REPORTS}/07_03_crime_type_push.csv", index=False)

    plot_data = crime_push.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(plot_data["Crime type"], plot_data["Push toward High severity"], height=0.65, color=BLUE)
    ax.axvline(0, color=INK_SOFT, linewidth=0.8)
    ax.grid(axis="x")
    ax.set_title("How each crime type moves the severity prediction")
    ax.set_xlabel("Push toward High severity (right) or Low severity (left), other columns held equal")

    save_chart(fig, "07_03_crime_type_push.png")
else:
    print("\nFinal model is not logistic regression, so crime type coefficients are skipped.")

print("\nSaved: reports/07_01_model_comparison.csv")
print("Saved: reports/07_02_feature_group_importance.csv")