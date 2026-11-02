# Seismic Activity Analytics

A Streamlit dashboard for exploring the USGS Comprehensive Earthquake Catalog (ComCat). It queries the public USGS GeoJSON API, caches results for one day, and computes the charts and analyses from the selected catalogue window.

## Run locally

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first query can take a little while because the app requests each calendar year separately. Use **Refresh USGS data** to bypass the cache. The default view covers 2000–2023 at magnitude 4.5 and is the same scope used in the project references.

## What the dashboard includes

- Epicentre map, magnitude and depth distributions, monthly counts, and UTC hour/month table.
- Gutenberg–Richter frequency–magnitude view and Aki maximum-likelihood b-value estimate.
- Monthly count dispersion and shallow/deep magnitude comparison.
- Isolation Forest anomaly flags with a downloadable event table.
- Chronological holdout comparison of naive, moving-average, and linear-trend monthly event-rate baselines.
- Global and India–Himalaya–Andaman catalogue windows, magnitude/depth filters, and CSV downloads.
- Upload a USGS CSV export to use the dashboard when the live API is unavailable.

The `M ≥ 7` metric in the anomalies tab is a proxy label for a limited comparison, not ground truth. Forecasts are event-rate baselines. They do not predict individual earthquakes and are not hazard warnings.

## Deploy on Streamlit Community Cloud

1. Keep `app.py`, `seismic.py`, `requirements.txt`, and `.streamlit/config.toml` in the repository.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with the GitHub account that can access the repository.
3. Choose **Create app**, select this repository and branch, set the main file to `app.py`, then deploy.

The app needs outbound access to `earthquake.usgs.gov`. No API key or Streamlit secrets are needed. The repository is public; create the Streamlit app in the account that owns it to get a hosted URL.

## Project structure

```text
app.py                  Streamlit interface
seismic.py              USGS fetch, cleaning, statistics, anomaly scoring, forecasting
requirements.txt        Runtime dependencies
.streamlit/config.toml   Streamlit theme and runtime settings
```

## Data and method notes

- Source: [USGS FDSN Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/).
- Requests are split into yearly windows to stay under the 20,000-event response cap.
- Essential numeric fields, UTC time, event type, and duplicate IDs are handled before analysis.
- Catalogue completeness varies across locations and time; the default b-value threshold is Mc 4.7.
- Aftershocks are not declustered. Isolation Forest flags are descriptive outliers, not precursors.
- The trend and moving-average baselines cannot anticipate unobserved earthquake sequences.
