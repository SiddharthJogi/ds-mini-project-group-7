from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st; import altair as alt

import seismic

st.set_page_config(page_title="Seismic Activity Analytics", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#172b3a; --muted:#71808d; --line:#e3e9ed; --blue:#126b82; --pale:#eff6f7; }
html, body, [class*="css"] { font-family:'Aptos','Segoe UI',sans-serif; }
.stApp { background:#f6f8f9; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(246,248,249,.9); }
[data-testid="stSidebar"] { background:#eef3f4; border-right:1px solid #e1e8ea; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { font-family:'Manrope',sans-serif; }
h1,h2,h3 { font-family:'Aptos Display','Segoe UI',sans-serif !important; letter-spacing:-.035em; color:#172b3a; }
.hero { padding:1.15rem 0 .5rem; }
.eyebrow { color:#126b82; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; }
.subcopy { color:#657684; max-width:850px; font-size:1rem; line-height:1.6; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:15px 17px; box-shadow:0 2px 8px rgba(20,48,65,.025); }
div[data-testid="stMetricLabel"] { color:#71808d; font-size:.78rem; }
div[data-testid="stMetricValue"] { color:#172b3a; font-family:'Aptos Display','Segoe UI',sans-serif; letter-spacing:-.04em; }
.note { border-left:3px solid #126b82; background:#edf5f6; padding:12px 16px; color:#415867; border-radius:0 8px 8px 0; margin:.8rem 0 1rem; }
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid #dfe6e9; }
.stTabs [data-baseweb="tab"] { padding:12px 2px; }
section[data-testid="stSidebar"] .stButton button { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def load_catalogue(start: date, end: date, minmag: float, region: str) -> pd.DataFrame:
    return seismic.fetch_catalogue(start, end, minmag, region)


@st.cache_data(show_spinner=False)
def score_catalogue(frame: pd.DataFrame, contamination: float) -> pd.DataFrame:
    return seismic.add_anomalies(frame, contamination)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


with st.sidebar:
    st.markdown("## Catalogue filters")
    source = st.radio("Data source", ["Live USGS API", "Upload USGS CSV"], horizontal=False)
    st.caption("USGS ComCat · keyless GeoJSON API")
    uploaded_file = st.file_uploader("USGS catalogue CSV", type=["csv"]) if source == "Upload USGS CSV" else None
    selected_dates = st.date_input("Event dates", value=(date(2000, 1, 1), date(2023, 12, 31)), min_value=date(1970, 1, 1), max_value=date.today())
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = date(2000, 1, 1), date(2023, 12, 31)
    minmag = st.slider("Minimum magnitude", min_value=2.5, max_value=7.0, value=4.5, step=0.1)
    depth_range = st.slider("Depth range (km)", min_value=0, max_value=700, value=(0, 700), step=10)
    region = st.radio("Geographic window", ["Global", "India-Himalaya-Andaman"], index=0)
    contamination = st.slider("Anomaly rate", 0.005, 0.05, 0.01, 0.005, format="%.3f")
    refresh = st.button("Refresh USGS data", use_container_width=True, type="primary")
    if refresh:
        load_catalogue.clear()
        score_catalogue.clear()
    st.markdown("---")
    st.caption("This catalogue summarizes recorded seismicity. It cannot predict the time, location, or magnitude of an individual earthquake.")

st.markdown('<div class="hero"><div class="eyebrow">USGS Comprehensive Earthquake Catalog</div><h1>Seismic activity analytics</h1><div class="subcopy">Explore earthquake patterns, compare statistical anomaly flags, and review simple monthly event-rate baselines. All results are calculated from the selected USGS catalogue window.</div></div>', unsafe_allow_html=True)

if start_date > end_date:
    st.error("Choose a start date that falls before the end date.")
    st.stop()

try:
    if source == "Upload USGS CSV":
        if uploaded_file is None:
            st.info("Upload a USGS ComCat CSV export to explore it without a live API connection.")
            st.stop()
        catalog = seismic.parse_csv_catalogue(uploaded_file)
        upper = pd.Timestamp(end_date + pd.Timedelta(days=1), tz="UTC")
        lower = pd.Timestamp(start_date, tz="UTC")
        raw = catalog.loc[catalog.time.ge(lower) & catalog.time.lt(upper) & catalog.mag.ge(minmag)].copy()
        if region == "India-Himalaya-Andaman":
            raw = raw.loc[raw.longitude.between(65, 100) & raw.latitude.between(5, 38)]
        st.caption("Using the uploaded CSV; records are filtered locally in this browser session.")
    else:
        with st.spinner("Retrieving the USGS catalogue in yearly slices…"):
            raw = load_catalogue(start_date, end_date, minmag, region)
except Exception as exc:
    st.error("The catalogue could not be loaded. Check the live USGS connection or confirm that the CSV has the required columns.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

if raw.empty:
    st.info("No earthquake events match these filters. Try widening the dates or lowering the magnitude threshold.")
    st.stop()

filtered = raw.loc[raw.depth.between(depth_range[0], depth_range[1])].copy()
if filtered.empty:
    st.info("No events remain in the selected depth range.")
    st.stop()

with st.spinner("Scoring unusual events…"):
    scored = score_catalogue(filtered, contamination)
events = scored
counts = seismic.monthly_counts(events)
bv, bse, b_n = seismic.b_value(events.mag, max(minmag, 4.7))
monthly_total = len(events) / max((end_date - start_date).days / 365.2425, 1 / 12)
shallow_pct = 100 * events.depth.lt(70).mean()

metrics = st.columns(6)
metrics[0].metric("Events in view", fmt_int(len(events)))
metrics[1].metric("Largest magnitude", f"M {events.mag.max():.1f}")
metrics[2].metric("Median depth", f"{events.depth.median():.0f} km")
metrics[3].metric("Shallow events", f"{shallow_pct:.0f}%")
metrics[4].metric(f"b-value · Mc {max(minmag, 4.7):.1f}", f"{bv:.2f}" if np.isfinite(bv) else "n/a", f"± {bse:.3f} SE · n={b_n:,}" if np.isfinite(bse) else None, delta_color="off")
metrics[5].metric("Anomalies flagged", fmt_int(events.anomaly.sum()), f"Isolation Forest · {contamination:.1%}", delta_color="off")

st.markdown('<div class="note"><b>Scope:</b> monthly forecasts estimate catalogue event rates. They are descriptive baselines and do not forecast individual earthquakes or provide hazard warnings.</div>', unsafe_allow_html=True)

tab_map, tab_time, tab_stats, tab_anom, tab_forecast, tab_about = st.tabs(["Epicentre map", "Time series", "Statistics", "Anomalies", "Forecast", "Data & limits"])

with tab_map:
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Where events were recorded")
        map_data = events[["latitude", "longitude"]].copy()
        if len(map_data) > 25000:
            map_data = map_data.sample(25000, random_state=42)
        st.map(map_data, latitude="latitude", longitude="longitude", size=8, color="#126b82", height=500)
        st.caption(f"Showing {len(map_data):,} of {len(events):,} events on the map when the full catalogue exceeds 25,000 rows.")
    with right:
        st.subheader("Magnitude distribution")
        hist = np.histogram(events.mag, bins=np.arange(np.floor(events.mag.min() * 2) / 2, events.mag.max() + 0.6, 0.2))
        hist_df = pd.DataFrame({"Magnitude bin": [f"{x:.1f}" for x in hist[1][:-1]], "Events": hist[0]})
        st.bar_chart(hist_df.set_index("Magnitude bin"), height=310, color="#126b82")
        st.subheader("Depth distribution")
        depth_df = events.assign(DepthBand=pd.cut(events.depth, [-1, 70, 300, 1000], labels=["Shallow <70 km", "Intermediate 70–300 km", "Deep >300 km"])).groupby("DepthBand", observed=False).size().rename("Events")
        st.bar_chart(depth_df, height=200, color="#e07b52")

with tab_time:
    st.subheader("Monthly earthquake event counts")
    st.caption(f"Events are grouped by UTC month · {monthly_total:,.0f} events per year on average across this selection.")
    st.line_chart(counts, height=360, color="#126b82")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Events by hour and month (UTC)**")
        heat = events.assign(Hour=events.time.dt.hour, Month=events.time.dt.month).pivot_table(index="Hour", columns="Month", values="id", aggfunc="count", fill_value=0).reindex(index=range(24), columns=range(1, 13), fill_value=0)
        heat.columns = [pd.Timestamp(2000, month, 1).strftime("%b") for month in heat.columns]
        st.dataframe(heat, use_container_width=True, height=420)
    with c2:
        st.markdown("**Largest monthly counts**")
        peaks = counts.sort_values(ascending=False).head(12).rename("Events").rename_axis("UTC month").reset_index()
        peaks["UTC month"] = peaks["UTC month"].dt.strftime("%Y-%m")
        st.dataframe(peaks, hide_index=True, use_container_width=True)

with tab_stats:
    st.subheader("Frequency–magnitude relationship")
    lcol, rcol = st.columns([1.2, 1])
    with lcol:
        step = 0.1
        mags = events.mag.dropna()
        bins = np.arange(max(minmag, np.floor(mags.min() * 10) / 10), mags.max() + step, step)
        if len(bins) < 2:
            bins = np.array([minmag, minmag + step])
        cumulative = np.array([(mags >= edge).sum() for edge in bins])
        gr = pd.DataFrame({"Magnitude threshold": bins, "Cumulative events": cumulative})
        gr = gr.loc[gr["Cumulative events"] > 0]
        st.altair_chart(alt.Chart(gr).mark_line(point=True, color="#126b82").encode(x=alt.X("Magnitude threshold:Q", title="Magnitude threshold"), y=alt.Y("Cumulative events:Q", scale=alt.Scale(type="log"), title="Cumulative events (log scale)"), tooltip=["Magnitude threshold:Q", "Cumulative events:Q"]).properties(height=350), use_container_width=True)
        st.caption("Gutenberg–Richter b-value uses the Aki maximum-likelihood estimator above Mc. The chart shows cumulative counts on a logarithmic scale.")
    with rcol:
        st.markdown("**Depth vs. magnitude**")
        scatter = events[["mag", "depth"]]
        if len(scatter) > 15000:
            scatter = scatter.sample(15000, random_state=42)
        st.scatter_chart(scatter, x="mag", y="depth", color="#126b82", size=18, height=350)
        tests = seismic.distribution_tests(events)
        st.markdown("**Count and magnitude checks**")
        st.write(f"Monthly variance-to-mean ratio: **{tests['dispersion']:.2f}**" if np.isfinite(tests["dispersion"]) else "Monthly dispersion unavailable.")
        st.write(f"Shallow vs. deep magnitude Mann–Whitney p-value: **{tests['mann_whitney_p']:.3g}**" if np.isfinite(tests["mann_whitney_p"]) else "Not enough shallow/deep events for a comparison.")
        st.caption("A dispersion ratio above 1 indicates counts vary more than a Poisson process with the same mean. Statistical significance does not imply practical importance.")

with tab_anom:
    st.subheader("Unusual events in the selected catalogue")
    st.caption("Isolation Forest scores combinations of magnitude, depth, and location. Flags are statistical outliers, not precursors or risk assessments.")
    anomalous = events.loc[events.anomaly].sort_values("anomaly_score", ascending=False)
    total = len(events)
    proxy = events.mag.ge(7.0)
    flagged_proxy = events.anomaly & proxy
    precision = flagged_proxy.sum() / max(events.anomaly.sum(), 1)
    recall = flagged_proxy.sum() / max(proxy.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    a, b, c, d = st.columns(4)
    a.metric("Events flagged", fmt_int(len(anomalous)))
    b.metric("M ≥ 7 proxy precision", f"{precision:.1%}")
    c.metric("M ≥ 7 proxy recall", f"{recall:.1%}")
    d.metric("Proxy F1", f"{f1:.3f}")
    st.caption(f"M ≥ 7.0 is only a high-magnitude proxy label for method comparison; it is not ground truth. It misses many other kinds of unusual seismicity. Total selected events: {total:,}.")
    show = anomalous[["time", "mag", "depth", "place", "anomaly_score", "sig", "alert"]].head(500).copy()
    show["time"] = show.time.dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(show, hide_index=True, use_container_width=True, height=480)
    st.download_button("Download flagged events (CSV)", anomalous.to_csv(index=False).encode("utf-8"), "usgs_anomalous_events.csv", "text/csv")

with tab_forecast:
    st.subheader("Monthly event-rate baselines")
    st.caption("Chronological holdout comparison. Forecasts describe counts per month; bursts associated with major earthquakes are difficult for simple baselines to anticipate.")
    scores, chart, future = seismic.forecast_comparison(counts)
    if scores.empty:
        st.info("Select at least 18 months of catalogue data to compare forecasting baselines.")
    else:
        st.line_chart(chart, height=390, color=["#126b82", "#e07b52", "#738895", "#8a9a6a"])
        st.dataframe(scores.round(2), hide_index=True, use_container_width=True)
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            st.metric("Next 12 months · baseline rate", f"{future.mean():.0f} events/month")
        with fcol2:
            st.info("This is a flat moving-average reference rate with no uncertainty interval. It is not a prediction of individual earthquakes or a warning system.")

with tab_about:
    st.subheader("Source, preprocessing, and interpretation")
    st.markdown(f"""
    **Source:** [USGS Earthquake Catalog (ComCat)](https://earthquake.usgs.gov/fdsnws/event/1/), queried directly as GeoJSON.

    **Selection:** {start_date.isoformat()} through {end_date.isoformat()}, magnitude ≥ {minmag:.1f}, region **{region}**, depth {depth_range[0]}–{depth_range[1]} km. The app requests one calendar year at a time to respect the API's event limit.

    **Cleaning:** GeoJSON properties and coordinates are flattened; UTC timestamps and numeric fields are parsed; non-earthquake events, missing essential coordinates/depth/magnitude, and duplicate event IDs are removed.

    **Methods:** b-value via Aki maximum likelihood above the chosen completeness threshold; monthly count dispersion and shallow/deep Mann–Whitney comparison; Isolation Forest anomaly scoring; naive, 12-month moving-average, and linear-trend forecast baselines with a chronological holdout.

    **Limits:** USGS catalogue coverage and magnitude completeness vary across place and time. Aftershocks are not declustered. The anomaly detector has no verified anomaly labels. The M ≥ 7 comparison is a proxy only. Forecast baselines do not account for tectonic context or individual event sequences. Earthquakes cannot be predicted from this dashboard.
    """)
    st.download_button("Download filtered catalogue (CSV)", events.to_csv(index=False).encode("utf-8"), "usgs_earthquakes.csv", "text/csv")

st.markdown("---")
st.caption(f"Data: USGS ComCat · {len(raw):,} catalogue records retrieved · Last app query: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} · Educational analysis only")
