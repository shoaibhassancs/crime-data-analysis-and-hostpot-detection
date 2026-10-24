import os
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"
REPORTS = f"{PROJECT_PATH}/reports"

os.makedirs(REPORTS, exist_ok=True)

pd.set_option("display.max_columns", None)  # show every column
pd.set_option("display.width", 200)         # keep tables on one line
pd.set_option("display.max_rows", 100)

# -----------------------------
# Load the Cleaned Data
# -----------------------------
df = pd.read_csv(CLEAN_FILE, parse_dates=["DATE"])

print("Rows and columns:", df.shape)

# -----------------------------
# Helper Functions
# -----------------------------
# Summary table: incidents, share of High severity and average severity per group
def summarise(data, group_col):
    table = data.groupby(group_col).agg(
        incidents=("INCIDENT_ID", "size"),
        pct_high=("SEVERITY", lambda s: round((s == "High").mean() * 100, 1)),
        avg_severity=("SEVERITY_SCORE", "mean"),
    )
    table["avg_severity"] = table["avg_severity"].round(2)
    return table

# Same table for time groups, but compared per day, because groups can have different numbers of days
def per_day_table(group_col):
    table = summarise(df, group_col)
    days_in_group = df.drop_duplicates("DATE").groupby(group_col).size()
    table["incidents_per_day"] = (table["incidents"] / days_in_group).round(1)
    table["vs_average_pct"] = ((table["incidents_per_day"] / overall_per_day - 1) * 100).round(1)
    return table

# -----------------------------
# Overall Picture
# -----------------------------
total_days = df["DATE"].nunique()
overall_per_day = len(df) / total_days

print("\nIncidents:", f"{len(df):,}")
print("Days covered:", total_days)
print("Average incidents per day:", round(overall_per_day, 1))

print("\nSeverity share (%):")
print((df["SEVERITY"].value_counts(normalize=True) * 100).round(1))

# -----------------------------
# Trends Over Time
# -----------------------------
# 12-month periods (Jun-May), so no year is cut short
period_table = per_day_table("PERIOD")
print("\nBy 12-month period (Jun-May):")
print(period_table)

month_table = per_day_table("MONTH")
print("\nBy calendar month:")
print(month_table)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_table = per_day_table("DAY_NAME").reindex(day_order)
print("\nBy day of the week:")
print(weekday_table)

# How big is the biggest difference from the average in each time view?
print("\nLargest gap from the average (% of average):")
print("Periods:", period_table["vs_average_pct"].abs().max())
print("Months:", month_table["vs_average_pct"].abs().max())
print("Weekdays:", weekday_table["vs_average_pct"].abs().max())

# -----------------------------
# Patterns by Location
# -----------------------------
town_table = summarise(df, "TOWN").sort_values("incidents", ascending=False)
print("\nBy town:")
print(town_table)

# Add town and risk level to the subdivision table
subdivision_table = summarise(df, "SUBDIVISION")
subdivision_table["town"] = df.groupby("SUBDIVISION")["TOWN"].first()
subdivision_table["risk_level"] = df.groupby("SUBDIVISION")["SUBDIVISION_RISK_LEVEL"].first()
subdivision_table = subdivision_table.sort_values("incidents", ascending=False)

print("\nTop 10 subdivisions by incidents:")
print(subdivision_table.head(10))

print("\nTop 10 subdivisions by average severity (200+ incidents only):")
print(subdivision_table[subdivision_table["incidents"] >= 200]
      .sort_values("avg_severity", ascending=False).head(10))

risk_order = ["Low", "Medium", "Medium-High", "High"]
risk_table = summarise(df, "SUBDIVISION_RISK_LEVEL").reindex(risk_order)
risk_table["subdivisions"] = df.groupby("SUBDIVISION_RISK_LEVEL")["SUBDIVISION"].nunique()
print("\nBy area risk level:")
print(risk_table)

# -----------------------------
# Patterns by Crime Type
# -----------------------------
crime_table = summarise(df, "CRIME_TYPE").sort_values("incidents", ascending=False)
print("\nBy crime type:")
print(crime_table)

# -----------------------------
# Outliers
# -----------------------------
# Unusual days: more than 3 standard deviations from the average daily count
daily = df.groupby("DATE").size()
upper_limit = daily.mean() + 3 * daily.std()
lower_limit = daily.mean() - 3 * daily.std()
unusual_days = daily[(daily > upper_limit) | (daily < lower_limit)]

print("\nDaily incidents:")
print(daily.describe().round(1))
print("\nUnusual days (outside 3 standard deviations):", len(unusual_days), "of", len(daily))
print(unusual_days)

print("\n5 busiest days:")
print(daily.sort_values(ascending=False).head(5))

# Unusual subdivisions: incident count above Q3 + 1.5 x IQR
q1 = subdivision_table["incidents"].quantile(0.25)
q3 = subdivision_table["incidents"].quantile(0.75)
subdivision_limit = q3 + 1.5 * (q3 - q1)
busy_subdivisions = subdivision_table[subdivision_table["incidents"] > subdivision_limit]

print("\nSubdivisions with unusually many incidents (above", round(subdivision_limit), "):",
      len(busy_subdivisions), "of", len(subdivision_table))
print(busy_subdivisions[["town", "risk_level", "incidents", "pct_high", "avg_severity"]])

# -----------------------------
# Save the Tables
# -----------------------------
period_table.to_csv(f"{REPORTS}/04_01_period_summary.csv")
month_table.to_csv(f"{REPORTS}/04_02_month_summary.csv")
weekday_table.to_csv(f"{REPORTS}/04_03_weekday_summary.csv")
town_table.to_csv(f"{REPORTS}/04_04_town_summary.csv")
subdivision_table.to_csv(f"{REPORTS}/04_05_subdivision_summary.csv")
risk_table.to_csv(f"{REPORTS}/04_06_risk_level_summary.csv")
crime_table.to_csv(f"{REPORTS}/04_07_crime_type_summary.csv")
unusual_days.rename("incidents").to_csv(f"{REPORTS}/04_08_unusual_days.csv")

print("\nSaved 8 tables in:", REPORTS)