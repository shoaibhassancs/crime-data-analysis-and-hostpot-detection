# Karachi Crime Incidents, Jun 2020 - May 2025: Findings Summary

**Question:** Where, when and what kind of incidents happen, and where should attention go first?

**Data:** 100,000 incident records across 18 towns, 194 subdivisions and 27 crime types, 1 Jun 2020 to 31 May 2025.
This is a synthetic dataset (the source column says "Synthetic"), so this is a demonstration of the method, not findings about real crime.

**Cleaning:** No missing values or duplicates. 5,607 labels were standardised ("Medium-High"), dates were converted from text, and 562 rows from one subdivision (Khokhrapar) have a coordinate far outside their own town. Those rows are kept in all counts and left off the map. Full details are in `cleaning_log.csv`.

## Key findings

**1. Nothing changes over time.** About 55 incidents per day, every year within 0.8% of the average, every month within 1.6%, every weekday within 1.0%. The share of High-severity incidents stays between 54.7% and 55.4% in every period. Day-to-day variation is what random chance alone would produce, and the 7 unusual days (out of 1,826) are about as many as chance predicts. There is no seasonal or weekday pattern to plan around.

**2. Where an incident happens matters most.** The 36 High-risk subdivisions (19% of all subdivisions) hold 40.7% of incidents, and 85.8% of those are High severity. In Low-risk areas only 9.6% are High severity. The same crime is also rated more severe in riskier areas (see `03_09_severity_heatmap.png`).

| Area risk level | Subdivisions | Incidents | High severity |
|---|---|---|---|
| Low | 82 | 18,832 | 9.6% |
| Medium | 68 | 34,891 | 43.5% |
| Medium-High | 8 | 5,607 | 57.2% |
| High | 36 | 40,670 | 85.8% |

**3. Two towns see the most incidents, two see the most severe ones.** Orangi (11,501) and Lyari (11,149) have the most incidents. Baldia (78.4% High) and Lyari (75.4%) have the most severe. North Nazimabad (9.7%) and Gulberg (15.3%) have the least.

**4. Twelve subdivisions stand out.** Each has more than 1,193 incidents, which makes them statistical outliers. All twelve are High-risk and together they account for 17.7% of all incidents. Six of them (Baghdadi, Kalakot, Landhi 89, Pirabad, Ghaziabad, Burns Road) are about 99% High severity.

**5. Crime types differ sharply.** Robbery (10,269) and Snatching (10,051) are the most frequent. Murder (96.2% High), Gang Violence (84.7%) and Homicide (84.1%) are the most severe. Industrial Hazards (3.5%) and Drug Possession (12.5%) are the least.

**6. Some crime labels are rated inconsistently.** Mobile Snatching is 0% High (all 124 are Medium), while Cellphone Snatching, Phone Theft and Street Snatching are 100% High. Overlapping labels (Murder and Homicide, Gang Violence and Gang Activity) could be merged. This is worth a definition check before any decisions rely on these categories.

## What this means

- Focus on place, not calendar: the 36 High-risk subdivisions, and the 12 standouts first.
- Do not build rosters or campaigns around months or weekdays, because the data shows no difference.
- Review the crime category definitions before relying on severity by crime type.

## Limits

- Synthetic data: results show the method, not real-world crime.
- Dates have no time of day, so hour-of-day analysis is not possible.
- Coordinates are neighbourhood-level (161 distinct points for 194 subdivisions), so maps show areas, not streets.
- Each subdivision has one fixed risk level, so the data cannot show risk changing over time.

*Charts: `visuals/03_01` to `03_09`. Tables: `reports/04_01` to `04_08`.*