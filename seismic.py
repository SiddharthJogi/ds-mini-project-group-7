"""USGS earthquake catalogue access and reproducible analysis helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2, mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
INDIA_BOUNDS = {"minlongitude": 65, "maxlongitude": 100, "minlatitude": 5, "maxlatitude": 38}
FIELDS = ["id", "time", "mag", "depth", "latitude", "longitude", "place", "type", "sig", "alert", "magType"]


def _parse_features(payload: dict[str, Any]) -> pd.DataFrame:
    records = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or [None, None, None]
        records.append({
            "id": feature.get("id"), "time": props.get("time"), "mag": props.get("mag"),
            "depth": coords[2] if len(coords) > 2 else None,
            "longitude": coords[0] if len(coords) > 0 else None,
            "latitude": coords[1] if len(coords) > 1 else None,
            "place": props.get("place"), "type": props.get("type"), "sig": props.get("sig"),
            "alert": props.get("alert"), "magType": props.get("magType"),
        })
    frame = pd.DataFrame.from_records(records, columns=FIELDS)
    if frame.empty:
        return frame
    frame["time"] = pd.to_datetime(frame["time"], unit="ms", utc=True, errors="coerce")
    for col in ("mag", "depth", "latitude", "longitude", "sig"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["time", "mag", "depth", "latitude", "longitude"])
    frame = frame.loc[(frame["type"] == "earthquake") & frame["mag"].between(-2, 10.5)]
    return frame.drop_duplicates("id").sort_values("time").reset_index(drop=True)


def parse_csv_catalogue(upload: Any) -> pd.DataFrame:
    """Normalize a USGS ComCat CSV export to the dashboard schema."""
    frame = pd.read_csv(upload)
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    required = {"time", "mag", "depth", "latitude", "longitude"}
    missing = required.difference(aliases)
    if missing:
        raise ValueError("CSV must include these USGS columns: time, mag, depth, latitude, longitude.")
    rename = {aliases[key]: key for key in required}
    for key in ("id", "place", "type", "sig", "alert", "magtype"):
        if key in aliases:
            rename[aliases[key]] = "magType" if key == "magtype" else key
    frame = frame.rename(columns=rename)
    if "id" not in frame:
        frame["id"] = [f"uploaded-{i}" for i in range(len(frame))]
    if "place" not in frame:
        frame["place"] = "Unknown location"
    if "type" not in frame:
        frame["type"] = "earthquake"
    for key in ("sig", "alert", "magType"):
        if key not in frame:
            frame[key] = np.nan
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    for col in ("mag", "depth", "latitude", "longitude", "sig"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["time", "mag", "depth", "latitude", "longitude"])
    frame = frame.loc[frame["type"].astype(str).str.lower().eq("earthquake")]
    return frame[FIELDS].drop_duplicates("id").sort_values("time").reset_index(drop=True)


def fetch_catalogue(start: date, end: date, min_magnitude: float, region: str) -> pd.DataFrame:
    """Fetch yearly slices so no USGS request exceeds its 20,000-event limit."""
    bounds = INDIA_BOUNDS if region == "India-Himalaya-Andaman" else {}
    pieces: list[pd.DataFrame] = []
    first_year, last_year = start.year, end.year
    session = requests.Session()
    session.headers.update({"User-Agent": "SeismicActivityAnalytics/1.0 (educational project)"})
    for year in range(first_year, last_year + 1):
        lo = max(start, date(year, 1, 1))
        hi = min(end + timedelta(days=1), date(year + 1, 1, 1))
        params = {
            "format": "geojson", "starttime": lo.isoformat(), "endtime": hi.isoformat(),
            "minmagnitude": min_magnitude, "orderby": "time-asc", "limit": 20000,
            "eventtype": "earthquake", **bounds,
        }
        response = session.get(USGS_URL, params=params, timeout=(10, 90))
        response.raise_for_status()
        payload = response.json()
        count = int((payload.get("metadata") or {}).get("count", 0))
        if count >= 20000:
            raise RuntimeError(f"USGS returned the 20,000-event limit for {year}. Raise the magnitude floor or use a smaller date range.")
        pieces.append(_parse_features(payload))
    nonempty = [part for part in pieces if not part.empty]
    if not nonempty:
        return pd.DataFrame(columns=FIELDS)
    data = pd.concat(nonempty, ignore_index=True).drop_duplicates("id")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    return data.loc[data.time.ge(start_ts) & data.time.lt(end_ts)].sort_values("time").reset_index(drop=True)


def add_anomalies(events: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Score unusual events without treating them as earthquake predictions."""
    out = events.copy()
    if out.empty:
        out["anomaly"] = pd.Series(dtype=bool)
        out["anomaly_score"] = pd.Series(dtype=float)
        return out
    features = out[["mag", "depth", "latitude", "longitude"]].astype(float)
    features = features.fillna(features.median())
    if len(out) < 20:
        out["anomaly"] = False
        out["anomaly_score"] = 0.0
        return out
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = IsolationForest(n_estimators=120, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(scaled)
    out["anomaly"] = model.predict(scaled) == -1
    out["anomaly_score"] = -model.score_samples(scaled)
    return out


def b_value(magnitudes: pd.Series, mc: float) -> tuple[float, float, int]:
    values = pd.to_numeric(magnitudes, errors="coerce").dropna().to_numpy()
    values = values[values >= mc]
    if len(values) < 2:
        return float("nan"), float("nan"), int(len(values))
    delta = 0.1
    denominator = float(np.mean(values) - (mc - delta / 2))
    b = float(np.log10(np.e) / denominator) if denominator > 0 else float("nan")
    se = float(2.30 * b * b / np.sqrt(len(values))) if np.isfinite(b) else float("nan")
    return b, se, int(len(values))


def monthly_counts(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=float, name="events")
    series = events.set_index("time").resample("MS").size().astype(float)
    return series.asfreq("MS", fill_value=0).rename("events")


def forecast_comparison(counts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Chronological last-fifth holdout, compared with three transparent baselines."""
    counts = counts.astype(float).asfreq("MS", fill_value=0)
    if len(counts) < 18:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
    split = max(12, int(len(counts) * 0.8))
    train, test = counts.iloc[:split], counts.iloc[split:]
    forecasts: dict[str, pd.Series] = {}
    forecasts["Naive (last month)"] = pd.Series(float(train.iloc[-1]), index=test.index)
    forecasts["Moving average (12)"] = pd.Series(float(train.tail(12).mean()), index=test.index)
    x_train = np.arange(len(train), dtype=float)
    slope, intercept = np.polyfit(x_train, train.to_numpy(), 1)
    forecasts["Linear trend"] = pd.Series(np.maximum(0, intercept + slope * np.arange(len(train), len(counts))), index=test.index)
    rows = []
    for name, prediction in forecasts.items():
        err = test.to_numpy() - prediction.to_numpy()
        rows.append({"Model": name, "MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err**2)),
                     "MAPE (%)": np.mean(np.abs(err) / np.maximum(test.to_numpy(), 1)) * 100})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    horizon = pd.date_range(counts.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future = pd.Series(float(counts.tail(12).mean()), index=horizon, name="12-month moving-average baseline")
    actual = pd.concat([train.rename("Train"), test.rename("Observed test")])
    chart = pd.DataFrame(index=actual.index.union(horizon))
    chart["Observed"] = actual
    chart["Moving average (12)"] = np.nan
    chart.loc[test.index, "Moving average (12)"] = forecasts["Moving average (12)"]
    chart.loc[test.index, "Linear trend"] = forecasts["Linear trend"]
    chart.loc[horizon, "12-month rate baseline"] = future
    return table, chart, future


def distribution_tests(events: pd.DataFrame) -> dict[str, float | int]:
    counts = monthly_counts(events)
    if len(counts) > 1 and counts.mean() > 0:
        dispersion = float(counts.var(ddof=1) / counts.mean())
        # Approximate two-sided over-dispersion screen against a Poisson variance.
        df = len(counts) - 1
        statistic = dispersion * df
        p_value = float(chi2.sf(statistic, df))
    else:
        dispersion, p_value = float("nan"), float("nan")
    shallow = events.loc[events.depth < 70, "mag"].dropna()
    deep = events.loc[events.depth >= 70, "mag"].dropna()
    if len(shallow) and len(deep):
        mw_p = float(mannwhitneyu(shallow, deep, alternative="two-sided").pvalue)
    else:
        mw_p = float("nan")
    return {"dispersion": dispersion, "poisson_p": p_value, "mann_whitney_p": mw_p,
            "shallow_n": int(len(shallow)), "deep_n": int(len(deep))}
