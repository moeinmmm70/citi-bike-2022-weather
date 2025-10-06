# CitiBike 2022 + Weather Analysis

This project explores how daily weather conditions in New York (LaGuardia Airport station) affect CitiBike usage during 2022.  
It combines CitiBike trip data with NOAA weather data and produces daily aggregated datasets for analysis and visualization.

---

## 📂 Project Structure

- [`data/`](data/)
  - [`raw/`](data/raw/) – Raw trip and weather files *(ignored in git; placeholder tracked as `.gitkeep`)*
  - [`processed/`](data/processed/) – Cleaned & merged datasets *(only key file tracked; placeholder `.gitkeep`)*
- [`notebooks/`](notebooks/) – Jupyter notebooks for downloading, cleaning, merging
- [`app/`](app/)
- [`reports/`](reports/)
  - [`map/`](map/)    – Visualizations and summary outputs
- [`README.md`](README.md) – Project documentation
- [`.gitignore`](.gitignore) – Ignore rules for large data

---

## 🚲 Data Sources

1. **CitiBike Trip Data (2022)**  
   - Downloaded from the official CitiBike data archive:  
     [https://s3.amazonaws.com/tripdata/index.html](https://s3.amazonaws.com/tripdata/index.html)  
   - Monthly files (e.g. `JC-202201-citibike-tripdata.csv.zip`) were pulled programmatically.  
   - Data includes ride IDs, start/end times, station info, bike type, and user type.

2. **NOAA Weather Data (LaGuardia Airport, NYC)**  
   - Retrieved using the NOAA Climate Data Online (CDO) API.  
   - Station ID: `USW00014732` (LaGuardia Airport).  
   - Daily average temperature (`TAVG`) was used.  
   - If `TAVG` was missing, it was estimated as `(TMIN + TMAX) / 2`.

---

## 🔗 Data Processing & Merge

1. **Bike Data**  
   - Each monthly ZIP file was read directly into pandas.  
   - Concatenated into a single DataFrame (outer join on columns to handle schema changes).  
   - Key columns normalized:
     - `started_at` parsed to datetime.
     - Trip durations converted to minutes.
     - Member type fields standardized.

2. **Weather Data**  
   - Queried daily temperatures for `2022-01-01` → `2022-12-31`.  
   - Cleaned into a tidy DataFrame with columns:
     - `date`
     - `avg_temp_c` (°C)

3. **Merge**  
   - Trips were aggregated to **daily totals**:
     - Number of rides
     - Average ride duration (min)
     - Share of members vs casual riders  
   - Merged with weather by `date`.

---

## 📊 Outputs

- **`data/processed/citibike_2022_daily_with_weather.csv`**  
  - ~365 rows (one per day in 2022).  
  - Columns:
    - `date`
    - `rides`
    - `avg_duration_min`
    - `member_share`
    - `avg_temp_c`

- **Notebooks**  
  - Scripts for downloading raw trip data.
  - Scripts for pulling NOAA weather data.
  - Data cleaning and merging workflows.

---

## 🚀 Next Steps

- Exploratory visualizations of ridership vs temperature.
- Time-series trends (seasonality, weekday vs weekend effects).
- Dashboard to interactively explore ridership and weather relationships.

---

## 🔑 Notes

- **Large raw data** (`data/raw/`) and intermediate files are excluded from Git to keep the repo light.  
- Only the key merged daily dataset is tracked in version control.  
- To reproduce, re-run the notebooks to download raw data and regenerate processed files.
