import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"
VISUALS = f"{PROJECT_PATH}/visuals"

os.makedirs(VISUALS, exist_ok=True)

# -----------------------------
# Load the Cleaned Data
# -----------------------------
df = pd.read_csv(CLEAN_FILE, parse_dates=["DATE"])

print("Rows and columns:", df.shape)

# -----------------------------
# Chart Style
# -----------------------------
# Colours: one blue family. Ordered values (Low -> High) go from light to dark
SURFACE = "#fcfcfb"   # chart background
INK = "#0b0b0b"       # main text
INK_SOFT = "#52514e"  # axis labels and notes
GRID = "#e4e3df"      # light grid lines
BLUE = "#2a78d6"      # single-colour charts

SEVERITY_ORDER = ["Low", "Medium", "High"]
SEVERITY_COLORS = {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#104281"}

RISK_ORDER = ["Low", "Medium", "Medium-High", "High"]
RISK_COLORS = {"Low": "#86b6ef", "Medium": "#3987e5", "Medium-High": "#1c5cab", "High": "#0d366b"}

# Light-to-dark blue scale for the map and the heatmap
BLUE_SCALE = LinearSegmentedColormap.from_list("blue_scale", ["#cde2fb", "#3987e5", "#0d366b"])

NOTE = "Synthetic data from Kaggle (Karachi crime), Jun 2020 - May 2025"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,  # grid lines are switched on per chart
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
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
# 03_01 Severity Distribution
# -----------------------------
severity_counts = df["SEVERITY"].value_counts().reindex(SEVERITY_ORDER)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(severity_counts.index, severity_counts.values, width=0.55,
              color=[SEVERITY_COLORS[s] for s in severity_counts.index])

# Label each bar with its count and share of all incidents
for bar, value in zip(bars, severity_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n({value / len(df):.1%})",
            ha="center", va="bottom")

ax.set_ylim(0, severity_counts.max() * 1.2)  # start at zero, leave room for labels
ax.grid(axis="y")
ax.set_title("Incidents by severity")
ax.set_ylabel("Number of incidents")

save_chart(fig, "03_01_severity_distribution.png")

# -----------------------------
# 03_02 Crime Types by Severity
# -----------------------------
crime_severity = pd.crosstab(df["CRIME_TYPE"], df["SEVERITY"])[SEVERITY_ORDER]
crime_severity = crime_severity.loc[crime_severity.sum(axis=1).sort_values().index]  # smallest at the bottom

fig, ax = plt.subplots(figsize=(10, 9))
crime_severity.plot(kind="barh", stacked=True, ax=ax, width=0.75,
                    color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                    edgecolor=SURFACE, linewidth=0.8)

ax.grid(axis="x")
ax.set_title("Incidents by crime type and severity")
ax.set_xlabel("Number of incidents")
ax.set_ylabel("")
ax.legend(title="Severity", frameon=False, loc="lower right")

save_chart(fig, "03_02_crime_types_by_severity.png")

# -----------------------------
# 03_03 Incidents by Town
# -----------------------------
town_counts = df["TOWN"].value_counts().sort_values()  # smallest at the bottom

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(town_counts.index, town_counts.values, height=0.65, color=BLUE)

for i, value in enumerate(town_counts.values):
    ax.text(value, i, f"  {value:,}", va="center", fontsize=9)

ax.set_xlim(0, town_counts.max() * 1.12)
ax.grid(axis="x")
ax.set_title("Incidents by town")
ax.set_xlabel("Number of incidents")

save_chart(fig, "03_03_incidents_by_town.png")

# -----------------------------
# 03_04 Top 15 Subdivisions
# -----------------------------
subdivision_summary = df.groupby("SUBDIVISION").agg(
    town=("TOWN", "first"),
    risk_level=("SUBDIVISION_RISK_LEVEL", "first"),
    incidents=("INCIDENT_ID", "size"),
    avg_severity=("SEVERITY_SCORE", "mean"),
).reset_index()

top15 = subdivision_summary.sort_values("incidents", ascending=False).head(15).iloc[::-1]  # biggest at the top
labels = top15["SUBDIVISION"] + " (" + top15["town"].str.replace(" Town", "") + ")"

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(labels, top15["incidents"], height=0.65,
        color=[RISK_COLORS[r] for r in top15["risk_level"]])

for i, value in enumerate(top15["incidents"].values):
    ax.text(value, i, f"  {value:,}", va="center", fontsize=9)

ax.set_xlim(0, top15["incidents"].max() * 1.12)
ax.grid(axis="x")
ax.set_title("Top 15 subdivisions by incident count")
ax.set_xlabel("Number of incidents")
ax.legend(handles=[Patch(color=RISK_COLORS[r], label=r) for r in RISK_ORDER],
          title="Area risk level", frameon=False, loc="lower right")

save_chart(fig, "03_04_top_subdivisions.png")

# -----------------------------
# 03_05 Incidents per Day, by Month
# -----------------------------
# Months have different lengths, so divide by the number of days in the month
monthly_counts = df.groupby("YEAR_MONTH").size()
month_dates = pd.to_datetime(monthly_counts.index + "-01")
incidents_per_day = monthly_counts.values / month_dates.days_in_month.values

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(month_dates, incidents_per_day, color=BLUE, linewidth=2)
ax.axhline(incidents_per_day.mean(), color=INK_SOFT, linestyle="--", linewidth=1,
           label=f"Average: {incidents_per_day.mean():.0f} per day")

ax.set_ylim(0, incidents_per_day.max() * 1.25)  # start at zero so a flat line looks flat
ax.grid(axis="y")
ax.set_title("Average incidents per day, by month")
ax.set_ylabel("Incidents per day")
ax.legend(frameon=False, loc="lower right")

save_chart(fig, "03_05_incidents_per_day_by_month.png")

# -----------------------------
# 03_06 Incidents by Day of the Week
# -----------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_counts = df["DAY_NAME"].value_counts().reindex(day_order)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(day_counts.index, day_counts.values, width=0.6, color=BLUE)

for bar, value in zip(bars, day_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

ax.set_ylim(0, day_counts.max() * 1.15)
ax.grid(axis="y")
ax.set_title("Incidents by day of the week")
ax.set_ylabel("Number of incidents")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

save_chart(fig, "03_06_incidents_by_weekday.png")

# -----------------------------
# 03_07 Severity Mix by Area Risk Level
# -----------------------------
# Share of Low / Medium / High incidents inside each risk level (each row adds up to 100%)
severity_mix = (pd.crosstab(df["SUBDIVISION_RISK_LEVEL"], df["SEVERITY"], normalize="index")
                .reindex(RISK_ORDER)[SEVERITY_ORDER] * 100)

fig, ax = plt.subplots(figsize=(9, 5))
left = np.zeros(len(severity_mix))

for severity in SEVERITY_ORDER:
    ax.barh(severity_mix.index, severity_mix[severity], left=left, height=0.65,
            color=SEVERITY_COLORS[severity], edgecolor=SURFACE, linewidth=1, label=severity)

    # Write the percentage inside the segment (skip segments that are too small)
    for i, value in enumerate(severity_mix[severity].values):
        if value >= 5:
            text_color = INK if severity == "Low" else "white"
            ax.text(left[i] + value / 2, i, f"{value:.0f}%", ha="center", va="center",
                    color=text_color, fontweight="bold")
    left = left + severity_mix[severity].values

ax.invert_yaxis()  # Low risk at the top
ax.set_xlim(0, 100)
ax.set_title("Severity mix by area risk level")
ax.set_xlabel("Share of incidents (%)")
ax.set_ylabel("Area risk level")
ax.legend(title="Severity", ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))

save_chart(fig, "03_07_severity_mix_by_risk_level.png")

# -----------------------------
# 03_08 Hotspot Map
# -----------------------------
# Leave out the rows with a suspicious coordinate (flagged during cleaning)
excluded = df[df["IS_SUSPECT_COORD"] == 1]
mapped = df[df["IS_SUSPECT_COORD"] == 0]

# Many incidents share one location, so group them into one bubble per location
locations = mapped.groupby(["LATITUDE", "LONGITUDE"]).agg(
    incidents=("INCIDENT_ID", "size"),
    avg_severity=("SEVERITY_SCORE", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 8))
points = ax.scatter(locations["LONGITUDE"], locations["LATITUDE"],
                    s=locations["incidents"] / 5, c=locations["avg_severity"], cmap=BLUE_SCALE,
                    vmin=1, vmax=10,  # same 1-10 scale as the heatmap
                    edgecolor=SURFACE, linewidth=1, alpha=0.9)

colorbar = fig.colorbar(points, ax=ax)
colorbar.set_label("Average severity score (1-10)")

ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Incident hotspots by location")
ax.set_xlabel("Longitude (bubble size = number of incidents)")
ax.set_ylabel("Latitude")

excluded_names = ", ".join(excluded["SUBDIVISION"].unique())
fig.text(0.01, 0.045, f"Not shown: {excluded_names} ({len(excluded):,} incidents), coordinate is far outside its town",
         fontsize=8, color=INK_SOFT, ha="left", va="bottom")

save_chart(fig, "03_08_hotspot_map.png")

# -----------------------------
# 03_09 Average Severity by Crime Type and Area Risk Level
# -----------------------------
severity_table = df.pivot_table(index="CRIME_TYPE", columns="SUBDIVISION_RISK_LEVEL",
                                values="SEVERITY_SCORE", aggfunc="mean")[RISK_ORDER]
severity_table = severity_table.loc[severity_table.mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(severity_table, annot=True, fmt=".1f", cmap=BLUE_SCALE, ax=ax,
            vmin=1, vmax=10,  # full 1-10 scale, so shades mean the same on every chart
            linewidths=1, linecolor=SURFACE, cbar_kws={"label": "Average severity score (1-10)"})

ax.set_title("Average severity by crime type and area risk level")
ax.set_xlabel("Area risk level (blank = that crime type never happens there)")
ax.set_ylabel("")

save_chart(fig, "03_09_severity_heatmap.png")

print("\nAll charts saved in:", VISUALS)