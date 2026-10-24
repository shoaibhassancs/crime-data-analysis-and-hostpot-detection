import os
import json
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
# Change these lines if your folder is somewhere else
PROJECT_PATH = "/Users/shoaibhassan/Desktop/PythonProjects/crime-data-analysis-and-hostpot-detection"
CLEAN_FILE = f"{PROJECT_PATH}/data/processed/karachi_crime_clean.csv"
DASHBOARD_FOLDER = f"{PROJECT_PATH}/dashboard"
DASHBOARD_FILE = f"{DASHBOARD_FOLDER}/08_dashboard.html"

os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

# -----------------------------
# Load the Cleaned Data
# -----------------------------
if not os.path.exists(CLEAN_FILE):
    raise FileNotFoundError(f"Clean data not found at: {CLEAN_FILE}. Run 02_data_cleaning.py first.")

df = pd.read_csv(CLEAN_FILE)
print("Rows and columns:", df.shape)

# The dashboard assumes each subdivision belongs to exactly one town
if df.groupby("SUBDIVISION")["TOWN"].nunique().max() > 1:
    raise ValueError("A subdivision name appears in more than one town. The dashboard needs one town per subdivision.")

# -----------------------------
# Prepare Lists the Dashboard Uses
# -----------------------------
# The dashboard does all the filtering inside the browser, so we send it
# one compact row per incident (numbers that point into the lists below).

severity_names = ["Low", "Medium", "High"]

towns = sorted(df["TOWN"].unique())
crime_types = sorted(df["CRIME_TYPE"].unique())
subdivisions = sorted(df["SUBDIVISION"].unique())

# Risk levels ordered from the safest to the most dangerous (by average severity score)
risk_levels = df.groupby("SUBDIVISION_RISK_LEVEL")["SEVERITY_SCORE"].mean().sort_values().index.tolist()

# Months: one number per month (0 = first month in the data)
first_month = df["YEAR"].min() * 12 + df.loc[df["YEAR"] == df["YEAR"].min(), "MONTH"].min()
df["MONTH_NUMBER"] = df["YEAR"] * 12 + df["MONTH"] - first_month

month_table = (
    df.groupby("MONTH_NUMBER")
    .agg(label=("YEAR_MONTH", "first"), period=("PERIOD", "first"))
    .sort_index()
)
periods = sorted(df["PERIOD"].unique())

# Days in each month, so we can show "incidents per day" (fair across 28 to 31 day months)
days_in_month = [int(pd.Period(label).days_in_month) for label in month_table["label"]]

# One row per subdivision: its town, risk level and map position
sub_table = df.groupby("SUBDIVISION").agg(
    town=("TOWN", "first"),
    risk=("SUBDIVISION_RISK_LEVEL", "first"),
    latitude=("LATITUDE", "mean"),
    longitude=("LONGITUDE", "mean"),
    suspect=("IS_SUSPECT_COORD", "mean"),
)

# Map points: subdivisions that share the same coordinates are drawn as one point.
# Subdivisions with wrong coordinates (flagged in step 02) are left off the map
map_subs = sub_table[sub_table["suspect"] < 0.5].copy()
map_subs["point_key"] = map_subs["latitude"].round(4).astype(str) + "," + map_subs["longitude"].round(4).astype(str)

point_keys = map_subs["point_key"].unique().tolist()
points = []
for key in point_keys:
    members = map_subs[map_subs["point_key"] == key]
    points.append({
        "lat": float(members["latitude"].iloc[0]),
        "lon": float(members["longitude"].iloc[0]),
        "names": members.index.tolist(),
    })

sub_info = []
for name in subdivisions:
    row = sub_table.loc[name]
    point = point_keys.index(map_subs.loc[name, "point_key"]) if name in map_subs.index else -1
    sub_info.append({
        "name": name,
        "town": towns.index(row["town"]),
        "risk": risk_levels.index(row["risk"]),
        "point": point,
    })

# -----------------------------
# Build the Data Package
# -----------------------------
data = {
    "severity": severity_names,
    "towns": towns,
    "crimeTypes": crime_types,
    "riskLevels": risk_levels,
    "periods": periods,
    "months": month_table["label"].tolist(),
    "monthPeriod": [periods.index(p) for p in month_table["period"]],
    "daysInMonth": days_in_month,
    "subdivisions": sub_info,
    "points": points,
    "rows": {
        "month": df["MONTH_NUMBER"].astype(int).tolist(),
        "weekday": df["DAY_OF_WEEK"].astype(int).tolist(),
        "sub": [subdivisions.index(s) for s in df["SUBDIVISION"]],
        "crime": [crime_types.index(c) for c in df["CRIME_TYPE"]],
        "severity": [severity_names.index(s) for s in df["SEVERITY"]],
        "score": df["SEVERITY_SCORE"].astype(int).tolist(),
    },
}

# -----------------------------
# Dashboard Page (HTML, CSS and JavaScript)
# -----------------------------
# The charts use the Chart.js library, loaded from the internet when the page opens
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karachi Crime Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --surface: #fcfcfb;
    --card: #ffffff;
    --ink: #0b0b0b;
    --ink-soft: #52514e;
    --line: #e4e3df;
    --blue: #2a78d6;
    --blue-dark: #104281;
    --blue-light: #86b6ef;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.45;
  }
  .page { max-width: 1240px; margin: 0 auto; padding: 24px 16px 40px; }
  h1 { margin: 0; font-size: 26px; }
  .subtitle { margin: 4px 0 20px; color: var(--ink-soft); }

  /* Filters */
  .filters {
    display: flex; flex-wrap: wrap; gap: 14px 18px; align-items: flex-end;
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 16px;
  }
  .filter { display: flex; flex-direction: column; gap: 4px; }
  .filter label, .filter .label { font-size: 12px; color: var(--ink-soft); }
  select {
    min-width: 160px; padding: 7px 8px; border: 1px solid var(--line); border-radius: 6px;
    background: var(--surface); color: var(--ink); font-size: 14px;
  }
  .chips { display: flex; gap: 6px; }
  .chip {
    display: flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 16px;
    border: 1px solid var(--line); background: var(--surface); color: var(--ink);
    font-size: 13px; cursor: pointer;
  }
  .chip .dot { width: 10px; height: 10px; border-radius: 50%; }
  .chip.off { opacity: 0.45; }
  .reset {
    padding: 7px 12px; border: 1px solid var(--line); border-radius: 6px;
    background: var(--surface); color: var(--ink); cursor: pointer; font-size: 13px;
  }
  .reset:hover, .chip:hover { border-color: var(--blue); }

  /* KPI tiles */
  .kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 16px; }
  .kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .kpi .value { font-size: 26px; font-weight: 700; }
  .kpi .value.small { font-size: 18px; padding-top: 6px; }
  .kpi .name { font-size: 12px; color: var(--ink-soft); }

  /* Chart cards */
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; min-width: 0;
          display: flex; flex-direction: column; }
  .card.wide { grid-column: 1 / -1; }
  .card h2 { margin: 0; font-size: 15px; }
  .card .hint { margin: 2px 0 8px; font-size: 12px; color: var(--ink-soft); }
  .box { position: relative; height: 300px; }
  .box.tall { height: 380px; }
  .box.map { height: auto; }
  .box.grow { height: auto; flex: 1; min-height: 300px; overflow: hidden; }

  .scale { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--ink-soft); margin-top: 6px; }
  .scale .bar { width: 160px; height: 10px; border-radius: 5px; background: linear-gradient(to right, #d7e6fa, #0b2a57); }

  .empty {
    display: none; background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 30px; text-align: center; color: var(--ink-soft); margin-bottom: 16px;
  }
  .note { margin-top: 20px; font-size: 12px; color: var(--ink-soft); }

  @media (max-width: 900px) {
    .kpis { grid-template-columns: repeat(2, 1fr); }
    .grid { grid-template-columns: 1fr; }
  }
  @media (max-width: 520px) {
    h1 { font-size: 22px; }
    select { min-width: 140px; }
    .box { height: 260px; }
    .box.tall { height: 420px; }
  }
</style>
</head>
<body>
<div class="page">
  <h1>Karachi Crime Dashboard</h1>
  <p class="subtitle" id="subtitle"></p>

  <div class="filters">
    <div class="filter">
      <label for="periodFilter">Year (Jun - May)</label>
      <select id="periodFilter"></select>
    </div>
    <div class="filter">
      <label for="townFilter">Town</label>
      <select id="townFilter"></select>
    </div>
    <div class="filter">
      <label for="crimeFilter">Crime type</label>
      <select id="crimeFilter"></select>
    </div>
    <div class="filter">
      <span class="label">Severity</span>
      <div class="chips" id="severityChips"></div>
    </div>
    <button class="reset" id="resetButton" type="button">Reset filters</button>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="value" id="kpiCount">-</div><div class="name">Incidents</div></div>
    <div class="kpi"><div class="value" id="kpiPerDay">-</div><div class="name">Incidents per day</div></div>
    <div class="kpi"><div class="value" id="kpiScore">-</div><div class="name">Average severity score (1-10)</div></div>
    <div class="kpi"><div class="value" id="kpiHigh">-</div><div class="name">High severity share</div></div>
    <div class="kpi"><div class="value small" id="kpiTop">-</div><div class="name">Busiest subdivision</div></div>
  </div>

  <div class="empty" id="emptyMessage">No incidents match these filters. Try resetting them.</div>

  <div class="grid" id="charts">
    <div class="card wide">
      <h2>Incidents per day, month by month</h2>
      <p class="hint">Monthly total divided by the days in that month, so short months are not unfairly low</p>
      <div class="box"><canvas id="trendChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Incidents by town</h2>
      <p class="hint">Click a bar to filter the whole dashboard to that town (click again to clear)</p>
      <div class="box tall"><canvas id="townChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Busiest subdivisions</h2>
      <p class="hint">Top 12 by number of incidents</p>
      <div class="box tall"><canvas id="subChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Crime types by severity</h2>
      <p class="hint">Top 12 crime types, split by severity</p>
      <div class="box tall"><canvas id="crimeChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Severity mix by area risk level</h2>
      <p class="hint">Share of each severity inside areas of each risk level</p>
      <div class="box tall"><canvas id="riskChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Incidents by weekday</h2>
      <p class="hint">Weekday patterns are close to flat in this data</p>
      <div class="box grow"><canvas id="weekdayChart"></canvas></div>
    </div>

    <div class="card">
      <h2>Hotspot map</h2>
      <p class="hint">Bigger circle = more incidents, darker = higher average severity. Subdivisions with wrong coordinates are left off</p>
      <div class="box map"><canvas id="mapChart"></canvas></div>
      <div class="scale"><span>Average severity 1</span><span class="bar"></span><span>10</span></div>
    </div>
  </div>

  <p class="note">
    Synthetic data from Kaggle (Karachi crime dataset), Jun 2020 - May 2025. Counts show how the dashboard works,
    not real crime levels in Karachi.
  </p>
</div>

<script>
const DATA = __DATA__;

// -----------------------------
// Colours and small helpers
// -----------------------------
const INK_SOFT = "#52514e";
const GRID = "#e4e3df";
const BLUE = "#2a78d6";
const BLUE_FADED = "#b9d3f3";
const SEVERITY_COLORS = ["#86b6ef", "#2a78d6", "#104281"];   // Low, Medium, High
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const rows = DATA.rows;
const rowCount = rows.month.length;
const subs = DATA.subdivisions;

// Blue scale for average severity (fixed 1-10, so colours mean the same in every view)
function scaleColor(score) {
  const t = Math.min(1, Math.max(0, (score - 1) / 9));
  const from = [215, 230, 250], to = [11, 42, 87];
  const c = from.map((v, i) => Math.round(v + (to[i] - v) * t));
  return "rgb(" + c.join(",") + ")";
}

function monthLabel(text, full) {      // "2020-06" -> "Jun 20" (or "Jun 2020" when full)
  const parts = text.split("-");
  return MONTH_NAMES[parseInt(parts[1], 10) - 1] + " " + (full ? parts[0] : parts[0].slice(2));
}

// Long labels are split over several lines (shorter lines on a phone)
function wrapLabel(text, maxChars) {
  if (text.length <= maxChars) return text;
  const lines = [];
  let line = "";
  text.split(" ").forEach(word => {
    if (line && (line + " " + word).length > maxChars) { lines.push(line); line = word; }
    else { line = line ? line + " " + word : word; }
  });
  if (line) lines.push(line);
  return lines.length > 1 ? lines : text;
}
const labelWidth = () => (window.innerWidth < 520 ? 15 : 34);

const format = n => n.toLocaleString("en-US");

// -----------------------------
// Filter state
// -----------------------------
const state = { period: -1, town: -1, crime: -1, severity: [true, true, true] };

function fillSelect(id, allText, items) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  const all = document.createElement("option");
  all.value = -1; all.textContent = allText;
  select.appendChild(all);
  items.forEach((text, i) => {
    const option = document.createElement("option");
    option.value = i; option.textContent = text;
    select.appendChild(option);
  });
  return select;
}

const periodSelect = fillSelect("periodFilter", "All years", DATA.periods);
const townSelect = fillSelect("townFilter", "All towns", DATA.towns);
const crimeSelect = fillSelect("crimeFilter", "All crime types", DATA.crimeTypes);

periodSelect.onchange = () => { state.period = +periodSelect.value; update(); };
townSelect.onchange = () => { state.town = +townSelect.value; update(); };
crimeSelect.onchange = () => { state.crime = +crimeSelect.value; update(); };

const chipBox = document.getElementById("severityChips");
DATA.severity.forEach((name, i) => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.innerHTML = '<span class="dot" style="background:' + SEVERITY_COLORS[i] + '"></span>' + name;
  chip.onclick = () => {
    state.severity[i] = !state.severity[i];
    chip.classList.toggle("off", !state.severity[i]);
    update();
  };
  chipBox.appendChild(chip);
});

document.getElementById("resetButton").onclick = () => {
  state.period = -1; state.town = -1; state.crime = -1; state.severity = [true, true, true];
  periodSelect.value = -1; townSelect.value = -1; crimeSelect.value = -1;
  chipBox.querySelectorAll(".chip").forEach(c => c.classList.remove("off"));
  update();
};

document.getElementById("subtitle").textContent =
  format(rowCount) + " incidents, " + monthLabel(DATA.months[0], true) + " to " +
  monthLabel(DATA.months[DATA.months.length - 1], true) + ". Use the filters to explore.";

// -----------------------------
// Chart.js setup
// -----------------------------
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
Chart.defaults.font.size = 12;
Chart.defaults.color = INK_SOFT;
Chart.defaults.animation = false;

// Writes a number at the end of each bar (and percentages inside stacked bars)
const labelPlugin = {
  id: "valueLabels",
  afterDatasetsDraw(chart) {
    const options = chart.options.plugins.valueLabels;
    if (!options) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.font = "11px " + Chart.defaults.font.family;

    if (options.texts) {
      const last = chart.data.datasets.length - 1;
      const meta = chart.getDatasetMeta(last);
      ctx.fillStyle = "#0b0b0b";
      meta.data.forEach((bar, i) => {
        if (!options.texts[i]) return;
        if (chart.options.indexAxis === "y") {
          ctx.textAlign = "left"; ctx.textBaseline = "middle";
          ctx.fillText(options.texts[i], bar.x + 5, bar.y);
        } else {
          ctx.textAlign = "center"; ctx.textBaseline = "bottom";
          ctx.fillText(options.texts[i], bar.x, bar.y - 4);
        }
      });
    }

    if (options.inside) {                       // percentages inside stacked bars
      chart.data.datasets.forEach((dataset, d) => {
        const meta = chart.getDatasetMeta(d);
        meta.data.forEach((bar, i) => {
          const width = Math.abs(bar.x - bar.base);
          const value = dataset.data[i];
          if (width < 34 || !value) return;
          ctx.fillStyle = d === 0 ? "#0b0b0b" : "#ffffff";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(Math.round(value) + "%", (bar.x + bar.base) / 2, bar.y);
        });
      });
    }
    ctx.restore();
  }
};
Chart.register(labelPlugin);

const gridLine = { color: GRID, lineWidth: 0.8 };
const noGrid = { display: false };

function makeBarOptions(horizontal, stacked) {
  const options = {
    indexAxis: horizontal ? "y" : "x",
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: { right: horizontal ? 36 : 0, top: horizontal ? 0 : 16 } },
    plugins: { legend: { display: false } },
    scales: {
      x: { stacked: stacked, grid: horizontal ? gridLine : noGrid, border: { display: false }, beginAtZero: true },
      y: { stacked: stacked, grid: horizontal ? noGrid : gridLine, border: { display: false }, beginAtZero: true },
    },
  };
  if (horizontal) {                    // show every label, wrapped when long
    options.scales.y.ticks = {
      autoSkip: false,
      callback: function (value) { return wrapLabel(this.getLabelForValue(value), labelWidth()); },
    };
  }
  return options;
}

const charts = {};

charts.trend = new Chart(document.getElementById("trendChart"), {
  type: "line",
  data: { labels: [], datasets: [{
    data: [], borderColor: BLUE, backgroundColor: BLUE, borderWidth: 2,
    pointRadius: 2, pointHoverRadius: 5, tension: 0.2,
  }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + " incidents per day" } },
    },
    scales: {
      x: { grid: noGrid, border: { color: GRID }, ticks: { maxTicksLimit: 12, maxRotation: 0 } },
      y: { beginAtZero: true, grid: gridLine, border: { display: false } },
    },
  },
});

charts.town = new Chart(document.getElementById("townChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ data: [], backgroundColor: BLUE, barThickness: 12 }] },
  options: Object.assign(makeBarOptions(true, false), {
    onClick: (event, elements) => {
      if (!elements.length) return;
      const index = elements[0].index;
      const townIndex = DATA.towns.indexOf(charts.town.data.labels[index]);
      state.town = state.town === townIndex ? -1 : townIndex;
      townSelect.value = state.town;
      update();
    },
    onHover: (event, elements) => { event.native.target.style.cursor = elements.length ? "pointer" : "default"; },
  }),
});

charts.sub = new Chart(document.getElementById("subChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ data: [], backgroundColor: BLUE, barThickness: 14 }] },
  options: makeBarOptions(true, false),
});

const crimeOptions = makeBarOptions(true, true);
crimeOptions.plugins.legend = { display: true, position: "top", align: "start", labels: { boxWidth: 10, boxHeight: 10 } };
charts.crime = new Chart(document.getElementById("crimeChart"), {
  type: "bar",
  data: { labels: [], datasets: DATA.severity.map((name, i) => ({
    label: name, data: [], backgroundColor: SEVERITY_COLORS[i], barThickness: 14,
  })) },
  options: crimeOptions,
});

const riskOptions = makeBarOptions(true, true);
riskOptions.plugins.legend = { display: true, position: "top", align: "start", labels: { boxWidth: 10, boxHeight: 10 } };
riskOptions.plugins.valueLabels = { inside: true };
riskOptions.scales.x.max = 100;
riskOptions.scales.x.ticks = { callback: v => v + "%" };
riskOptions.layout.padding.right = 8;
riskOptions.plugins.tooltip = { callbacks: { label: ctx => ctx.dataset.label + ": " + ctx.parsed.x.toFixed(1) + "%" } };
charts.risk = new Chart(document.getElementById("riskChart"), {
  type: "bar",
  data: { labels: DATA.riskLevels.map(r => r + " risk"), datasets: DATA.severity.map((name, i) => ({
    label: name, data: [], backgroundColor: SEVERITY_COLORS[i], barThickness: 34,
  })) },
  options: riskOptions,
});

charts.weekday = new Chart(document.getElementById("weekdayChart"), {
  type: "bar",
  data: { labels: WEEKDAYS, datasets: [{ data: [], backgroundColor: BLUE, barPercentage: 0.6 }] },
  options: makeBarOptions(false, false),
});

// Map: longitude across, latitude up, with the aspect ratio of the area itself
const lons = DATA.points.map(p => p.lon), lats = DATA.points.map(p => p.lat);
const lonMin = Math.min(...lons) - 0.03, lonMax = Math.max(...lons) + 0.03;
const latMin = Math.min(...lats) - 0.03, latMax = Math.max(...lats) + 0.03;
const mapRatio = Math.min(1.9, Math.max(1.0,
  ((lonMax - lonMin) * Math.cos(24.9 * Math.PI / 180)) / (latMax - latMin)));

charts.map = new Chart(document.getElementById("mapChart"), {
  type: "bubble",
  data: { datasets: [{ data: [], backgroundColor: [], borderColor: "#ffffff", borderWidth: 0.8 }] },
  options: {
    responsive: true, maintainAspectRatio: true, aspectRatio: mapRatio,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: items => {
            const names = items[0].raw.names;
            return names.length > 3 ? names.slice(0, 3).join(", ") + " + " + (names.length - 3) + " more" : names.join(", ");
          },
          label: ctx => [
            format(ctx.raw.count) + " incidents",
            "Average severity " + ctx.raw.average.toFixed(1) + " out of 10",
          ],
        },
      },
    },
    scales: {
      x: { min: lonMin, max: lonMax, grid: gridLine, border: { display: false }, title: { display: true, text: "Longitude" } },
      y: { min: latMin, max: latMax, grid: gridLine, border: { display: false }, title: { display: true, text: "Latitude" } },
    },
  },
});

// -----------------------------
// Recalculate everything for the current filters
// -----------------------------
function update() {
  const monthCount = new Array(DATA.months.length).fill(0);
  const weekdayCount = new Array(7).fill(0);
  const townCount = new Array(DATA.towns.length).fill(0);
  const subCount = new Array(subs.length).fill(0);
  const crimeCount = DATA.crimeTypes.map(() => [0, 0, 0]);
  const riskCount = DATA.riskLevels.map(() => [0, 0, 0]);
  const pointCount = new Array(DATA.points.length).fill(0);
  const pointScore = new Array(DATA.points.length).fill(0);

  let total = 0, scoreSum = 0, highCount = 0;

  for (let i = 0; i < rowCount; i++) {
    const sev = rows.severity[i];
    if (!state.severity[sev]) continue;
    const month = rows.month[i];
    if (state.period !== -1 && DATA.monthPeriod[month] !== state.period) continue;
    const crime = rows.crime[i];
    if (state.crime !== -1 && crime !== state.crime) continue;

    const sub = rows.sub[i];
    const town = subs[sub].town;

    townCount[town]++;                              // the town chart ignores the town filter
    if (state.town !== -1 && town !== state.town) continue;

    total++;
    scoreSum += rows.score[i];
    if (sev === 2) highCount++;
    monthCount[month]++;
    weekdayCount[rows.weekday[i]]++;
    subCount[sub]++;
    crimeCount[crime][sev]++;
    riskCount[subs[sub].risk][sev]++;
    const point = subs[sub].point;
    if (point >= 0) { pointCount[point]++; pointScore[point] += rows.score[i]; }
  }

  document.getElementById("emptyMessage").style.display = total === 0 ? "block" : "none";
  document.getElementById("charts").style.display = total === 0 ? "none" : "grid";

  // Which months are in view? (needed for incidents per day)
  const monthsInView = [];
  DATA.months.forEach((label, m) => {
    if (state.period === -1 || DATA.monthPeriod[m] === state.period) monthsInView.push(m);
  });
  const daysInView = monthsInView.reduce((sum, m) => sum + DATA.daysInMonth[m], 0);

  // KPI tiles
  let topIndex = -1;
  subCount.forEach((c, i) => { if (c > 0 && (topIndex === -1 || c > subCount[topIndex])) topIndex = i; });

  document.getElementById("kpiCount").textContent = format(total);
  document.getElementById("kpiPerDay").textContent = total ? (total / daysInView).toFixed(1) : "-";
  document.getElementById("kpiScore").textContent = total ? (scoreSum / total).toFixed(1) : "-";
  document.getElementById("kpiHigh").textContent = total ? (100 * highCount / total).toFixed(0) + "%" : "-";
  document.getElementById("kpiTop").textContent = topIndex >= 0 ? subs[topIndex].name + " (" + format(subCount[topIndex]) + ")" : "-";

  // Trend
  charts.trend.data.labels = monthsInView.map(m => monthLabel(DATA.months[m]));
  charts.trend.data.datasets[0].data = monthsInView.map(m => monthCount[m] / DATA.daysInMonth[m]);
  charts.trend.update();

  // Towns (all towns, sorted, selected town highlighted)
  const townOrder = DATA.towns.map((name, i) => i).sort((a, b) => townCount[b] - townCount[a]);
  charts.town.data.labels = townOrder.map(i => DATA.towns[i]);
  charts.town.data.datasets[0].data = townOrder.map(i => townCount[i]);
  charts.town.data.datasets[0].backgroundColor = townOrder.map(i =>
    state.town === -1 || state.town === i ? BLUE : BLUE_FADED);
  charts.town.options.plugins.valueLabels = { texts: townOrder.map(i => format(townCount[i])) };
  charts.town.update();

  // Subdivisions (top 12)
  const subOrder = subs.map((s, i) => i).filter(i => subCount[i] > 0)
    .sort((a, b) => subCount[b] - subCount[a]).slice(0, 12);
  charts.sub.data.labels = subOrder.map(i => subs[i].name + " (" + DATA.riskLevels[subs[i].risk] + ")");
  charts.sub.data.datasets[0].data = subOrder.map(i => subCount[i]);
  charts.sub.options.plugins.valueLabels = { texts: subOrder.map(i => format(subCount[i])) };
  charts.sub.update();

  // Crime types (top 12)
  const crimeTotal = crimeCount.map(c => c[0] + c[1] + c[2]);
  const crimeOrder = DATA.crimeTypes.map((n, i) => i).filter(i => crimeTotal[i] > 0)
    .sort((a, b) => crimeTotal[b] - crimeTotal[a]).slice(0, 12);
  charts.crime.data.labels = crimeOrder.map(i => DATA.crimeTypes[i]);
  DATA.severity.forEach((name, s) => {
    charts.crime.data.datasets[s].data = crimeOrder.map(i => crimeCount[i][s]);
  });
  charts.crime.options.plugins.valueLabels = { texts: crimeOrder.map(i => format(crimeTotal[i])) };
  charts.crime.update();

  // Severity mix by risk level (percent of each risk level)
  DATA.severity.forEach((name, s) => {
    charts.risk.data.datasets[s].data = riskCount.map(c => {
      const sum = c[0] + c[1] + c[2];
      return sum ? 100 * c[s] / sum : 0;
    });
  });
  charts.risk.update();

  // Weekday
  charts.weekday.data.datasets[0].data = weekdayCount;
  charts.weekday.options.plugins.valueLabels = { texts: weekdayCount.map(c => format(c)) };
  charts.weekday.update();

  // Map
  const mapPoints = [], mapColors = [];
  const maxCount = Math.max(1, ...pointCount);
  pointCount.forEach((count, i) => {
    if (count === 0) return;
    const average = pointScore[i] / count;
    mapPoints.push({
      x: DATA.points[i].lon, y: DATA.points[i].lat,
      r: 3 + 14 * Math.sqrt(count / maxCount),
      count: count, average: average, names: DATA.points[i].names,
    });
    mapColors.push(scaleColor(average));
  });
  charts.map.data.datasets[0].data = mapPoints;
  charts.map.data.datasets[0].backgroundColor = mapColors;
  charts.map.update();
}

let resizeTimer;
window.addEventListener("resize", () => {   // re-wrap the labels when the window size changes
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(update, 200);
});

update();
</script>
</body>
</html>
"""

# -----------------------------
# Save the Dashboard
# -----------------------------
html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))

with open(DASHBOARD_FILE, "w", encoding="utf-8") as file:
    file.write(html)

size_mb = os.path.getsize(DASHBOARD_FILE) / 1_000_000
print(f"Towns: {len(towns)} | Subdivisions: {len(subdivisions)} | Crime types: {len(crime_types)}")
print(f"Map points: {len(points)} | Months: {len(month_table)} | Periods: {periods}")
print(f"\nSaved: {DASHBOARD_FILE} ({size_mb:.1f} MB)")
print("Open it by double-clicking the file (it needs an internet connection to load the chart library).")