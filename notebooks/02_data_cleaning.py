import os
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder or file is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
RAW_FILE = f"{PROJECT_PATH}/data/raw/karachi_crime_2020_2025.csv"
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"
LOG_FILE = f"{PROJECT_PATH}/reports/cleaning_log.csv"

pd.set_option("display.max_columns", None)   # show every column
pd.set_option("display.width", 250)          # keep tables on one line
pd.set_option("display.max_colwidth", None)  # do not cut long notes

# Make sure the output folders exist
os.makedirs(f"{PROJECT_PATH}/data/processed", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/reports", exist_ok=True)

# -----------------------------
# Load the Dataset
# -----------------------------
if not os.path.exists(RAW_FILE):
    raise FileNotFoundError(f"Raw data not found at: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print("Rows and columns before cleaning:", df.shape)

# -----------------------------
# Cleaning Log
# -----------------------------
# Every step is written to this list, so we can show exactly what changed and why.
# "rows" and "columns" are the counts AFTER the step.
cleaning_log = []

def log_step(action, notes):
    cleaning_log.append({
        "step": len(cleaning_log) + 1,
        "action": action,
        "rows": len(df),
        "columns": df.shape[1],
        "notes": notes,
    })

log_step("Load raw data", "karachi_crime_2020_2025.csv, unchanged")

# -----------------------------
# Handle Missing Values
# -----------------------------
missing_cells = int(df.isnull().sum().sum())
rows_before = len(df)

df = df.dropna()  # remove rows with missing values

log_step("Remove rows with missing values",
         f"{missing_cells} missing cells found, {rows_before - len(df)} rows removed")

# -----------------------------
# Handle Duplicates
# -----------------------------
duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["INCIDENT_ID"].duplicated().sum())
rows_before = len(df)

df = df.drop_duplicates()  # remove fully duplicated rows

log_step("Remove duplicate rows",
         f"{duplicate_rows} duplicate rows and {duplicate_ids} duplicate INCIDENT_IDs found, "
         f"{rows_before - len(df)} rows removed")

# -----------------------------
# Standardise Text Values
# -----------------------------
# Find every text column, remove extra spaces and replace the long dash
# in labels such as 'Medium–High' with a normal hyphen
text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]

spaces_fixed = 0
dashes_fixed = 0

for col in text_cols:
    spaces_fixed += int((df[col] != df[col].str.strip()).sum())
    df[col] = df[col].str.strip()

    dashes_fixed += int(df[col].str.contains("–", regex=False).sum())
    df[col] = df[col].str.replace("–", "-", regex=False)

log_step("Standardise text values",
         f"{spaces_fixed} values had extra spaces, {dashes_fixed} values had a long dash "
         f"(e.g. 'Medium–High' became 'Medium-High')")

# -----------------------------
# Correct Data Types
# -----------------------------
# DATE is stored as text, so change it to a proper date
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")

log_step("Convert DATE from text to date",
         f"Range {df['DATE'].min().date()} to {df['DATE'].max().date()}. "
         f"The data has dates only, no time of day")

# -----------------------------
# Check Consistency (nothing is changed here)
# -----------------------------
# 1. SEVERITY must match SEVERITY_SCORE (1-4 Low, 5-7 Medium, 8-10 High)
severity_ok = (
    ((df["SEVERITY_SCORE"] <= 4) & (df["SEVERITY"] == "Low"))
    | (df["SEVERITY_SCORE"].between(5, 7) & (df["SEVERITY"] == "Medium"))
    | ((df["SEVERITY_SCORE"] >= 8) & (df["SEVERITY"] == "High"))
)
severity_mismatch = int((~severity_ok).sum())

log_step("Check SEVERITY matches SEVERITY_SCORE",
         f"{severity_mismatch} mismatches. SEVERITY_SCORE is SEVERITY in numbers, "
         f"so it must not be used to predict SEVERITY")

# 2. The IS_*_ZONE flags should just repeat RISK_ZONE
zone_flags = {
    "IS_RED_ZONE": "Red",
    "IS_ORANGE_ZONE": "Orange",
    "IS_YELLOW_ZONE": "Yellow",
    "IS_GREEN_ZONE": "Green",
}
flags_match = all(
    ((df[flag] == 1) == (df["RISK_ZONE"] == zone)).all()
    for flag, zone in zone_flags.items()
)

log_step("Check IS_*_ZONE flags against RISK_ZONE",
         f"Flags are exact copies of RISK_ZONE: {flags_match}")

# 3. RISK_ZONE and SUBDIVISION_RISK_LEVEL should carry the same information
levels_per_zone = df.groupby("RISK_ZONE")["SUBDIVISION_RISK_LEVEL"].nunique().max()
zones_per_level = df.groupby("SUBDIVISION_RISK_LEVEL")["RISK_ZONE"].nunique().max()
same_information = bool(levels_per_zone == 1 and zones_per_level == 1)

log_step("Check RISK_ZONE against SUBDIVISION_RISK_LEVEL",
         f"One-to-one match (same information, different labels): {same_information}")

# -----------------------------
# Flag Suspicious Coordinates
# -----------------------------
# Find the centre of each town (the middle of its subdivision coordinates)
sub_coords = df.groupby(["TOWN", "SUBDIVISION"])[["LATITUDE", "LONGITUDE"]].first()
town_centre = sub_coords.groupby("TOWN")[["LATITUDE", "LONGITUDE"]].median()

# How far is each incident from the centre of its own town? (in degrees, 0.1 is about 11 km)
lat_gap = (df["LATITUDE"] - df["TOWN"].map(town_centre["LATITUDE"])).abs()
lon_gap = (df["LONGITUDE"] - df["TOWN"].map(town_centre["LONGITUDE"])).abs()

# More than 0.2 degrees (about 22 km) from its own town looks like a wrong coordinate.
# We flag these rows instead of deleting them, because the crime records themselves are fine
suspect = (lat_gap > 0.2) | (lon_gap > 0.2)
df["IS_SUSPECT_COORD"] = suspect.astype(int)

suspect_places = df.loc[suspect, "SUBDIVISION"].unique().tolist()

log_step("Flag suspicious coordinates",
         f"{int(suspect.sum())} rows flagged in IS_SUSPECT_COORD (subdivision: {suspect_places}), "
         f"because they are more than 0.2 degrees from their own town. "
         f"Rows are kept for counts, but should be left off maps")

# Some coordinates are rounded to 2 decimals (e.g. 24.80, 66.99). If they sit inside
# their own town they are valid, just less precise, so they are kept and not flagged
rounded = (df["LATITUDE"].round(2) == df["LATITUDE"]) & (df["LONGITUDE"].round(2) == df["LONGITUDE"])
rounded_but_valid = df.loc[rounded & ~suspect, "SUBDIVISION"].unique().tolist()

log_step("Note rounded coordinates",
         f"Subdivisions with coordinates rounded to 2 decimals but inside their own town: "
         f"{rounded_but_valid}. Kept as valid")

# -----------------------------
# Drop Unneeded Columns
# -----------------------------
# Columns with a single value carry no information
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
source_values = df["SOURCE"].unique().tolist()

df = df.drop(columns=constant_cols)

log_step("Drop columns with a single value",
         f"Dropped {constant_cols}. SOURCE was {source_values}, so this is synthetic data")

# The zone flags repeat RISK_ZONE, so drop them (only if the check above passed)
if flags_match:
    df = df.drop(columns=list(zone_flags))
    log_step("Drop IS_*_ZONE flags", f"Dropped {list(zone_flags)}, RISK_ZONE keeps the same information")

# -----------------------------
# Feature Engineering
# -----------------------------
# Derive date features. There is no HOUR, because DATE has no time of day
df["YEAR"] = df["DATE"].dt.year
df["MONTH"] = df["DATE"].dt.month
df["YEAR_MONTH"] = df["DATE"].dt.strftime("%Y-%m")
df["DAY_OF_WEEK"] = df["DATE"].dt.day_of_week  # Monday = 0, Sunday = 6
df["DAY_NAME"] = df["DATE"].dt.day_name()
df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)  # Saturday and Sunday

# The data runs June to May, so group by 12-month periods (Jun-May)
# to avoid partial years at both ends (2020 and 2025)
period_start = df["YEAR"].where(df["MONTH"] >= 6, df["YEAR"] - 1)
df["PERIOD"] = period_start.astype(str) + "/" + ((period_start + 1) % 100).astype(str).str.zfill(2)

log_step("Add date features",
         "Added YEAR, MONTH, YEAR_MONTH, DAY_OF_WEEK, DAY_NAME, IS_WEEKEND and PERIOD (Jun-May year, e.g. 2020/21)")

# -----------------------------
# Save the Cleaned Data and the Cleaning Log
# -----------------------------
df.to_csv(CLEAN_FILE, index=False)

log_df = pd.DataFrame(cleaning_log)
log_df.to_csv(LOG_FILE, index=False)

print(log_df.to_string(index=False))

print("\nRows and columns after cleaning:", df.shape)
print("\nColumns:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head())
print("\nSaved:", CLEAN_FILE)
print("Saved:", LOG_FILE)