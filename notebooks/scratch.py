import pandas as pd

# -----------------------------
# Settings
# -----------------------------
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

df = pd.read_csv(CLEAN_FILE)

# -----------------------------
# Flagged subdivisions
# -----------------------------
flagged = df[df["IS_SUSPECT_COORD"] == 1]

print("Flagged subdivisions and their coordinates:")
print(flagged.groupby(["TOWN", "SUBDIVISION"]).agg(
    rows=("INCIDENT_ID", "size"),
    lat=("LATITUDE", "first"),
    lon=("LONGITUDE", "first"),
))

# -----------------------------
# Other subdivisions in the same towns
# -----------------------------
towns = flagged["TOWN"].unique()

print("\nAll subdivisions in those towns:")
print(df[df["TOWN"].isin(towns)].groupby(["TOWN", "SUBDIVISION"]).agg(
    rows=("INCIDENT_ID", "size"),
    lat=("LATITUDE", "first"),
    lon=("LONGITUDE", "first"),
))

# -----------------------------
# Check that SEVERITY and SEVERITY_SCORE look right in the saved file
# -----------------------------
print("\nSeverity labels in the saved file:", df["SEVERITY"].unique().tolist())
print(df[["SEVERITY_SCORE", "SEVERITY"]].head())