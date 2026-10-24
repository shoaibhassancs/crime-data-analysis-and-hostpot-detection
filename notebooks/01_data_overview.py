import os
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder or file is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
RAW_FILE = f"{PROJECT_PATH}/data/raw/karachi_crime_2020_2025.csv"

pd.set_option("display.max_columns", None)  # show every column
pd.set_option("display.width", 200)         # keep tables on one line

# -----------------------------
# Load the Dataset
# -----------------------------
# Stop with a clear message if the file is not where we expect it
if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"Raw data not found at: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print("Rows and columns:", df.shape)

# -----------------------------
# First Look at the Data
# -----------------------------
print(df.head())
print()
df.info()  # column names, data types and non-null counts

# -----------------------------
# Check Missing Values & Duplicates
# -----------------------------
print("Missing values per column:")
print(df.isnull().sum())

print("\nFully duplicated rows:", df.duplicated().sum())
print("Duplicated INCIDENT_ID:", df["INCIDENT_ID"].duplicated().sum())

# -----------------------------
# Check Categorical Columns
# -----------------------------
# How many different values does each column have?
print("Unique values per column:")
print(df.nunique())

# Columns with only 1 value carry no information
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("\nColumns with a single value:", constant_cols)

print("\nSeverity:")
print(df["SEVERITY"].value_counts())
print(df["SEVERITY"].value_counts(normalize=True).round(3))  # shares of 1.0

print("\nCrime types:")
print(df["CRIME_TYPE"].value_counts())

print("\nTowns:")
print(df["TOWN"].value_counts())

print("\nTop 10 subdivisions:")
print(df["SUBDIVISION"].value_counts().head(10))

# -----------------------------
# Check Numeric Columns
# -----------------------------
print(df[["SEVERITY_SCORE", "LATITUDE", "LONGITUDE"]].describe())

# -----------------------------
# Check the Date Column
# -----------------------------
# DATE is stored as text, so look at a few values first
print("Sample values:", df["DATE"].head().tolist())

# Convert a temporary copy to dates just to see the range
dates = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
print("First date:", dates.min().date())
print("Last date:", dates.max().date())
print("Days covered:", dates.nunique())

# Does the date include a time of day?
print("Unique hours in DATE:", dates.dt.hour.nunique())