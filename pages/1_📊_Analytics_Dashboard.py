"""
Analytics Dashboard — Water Body Analysis

A read-only dashboard that visualises historical water-body analyses stored
in the project's SQLite database.  No model loading is required — this page
only reads from the DB.

This file lives in  project/pages/  and is auto-discovered by Streamlit's
multi-page feature.  It does NOT modify app.py or any existing file.
"""

import os
import sys
import datetime

# ---------------------------------------------------------------------------
# Ensure the parent directory (project root) is on sys.path so we can
# import config, database, etc. without duplicating them.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

# Reuse existing project modules — no duplication
import config
import database

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Dashboard — Water Body Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Initialise DB (idempotent — safe to call multiple times)
# ---------------------------------------------------------------------------
database.init_database()

# ---------------------------------------------------------------------------
# Scoped CSS — mirrors the existing dark-teal theme from app.py
# without changing global styles.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Main background */
  .stApp { background: #0d1117; color: #e2e8f0; }

  /* ── Dashboard-specific styles ────────────────────────────────── */

  /* KPI cards row */
  .kpi-card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(6,182,212,0.25);
      border-radius: 16px;
      padding: 22px 20px 18px;
      text-align: center;
      transition: transform .18s, box-shadow .18s;
  }
  .kpi-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 8px 28px rgba(6,182,212,0.18);
  }
  .kpi-icon  { font-size: 1.8rem; margin-bottom: 4px; }
  .kpi-value {
      font-size: 1.55rem; font-weight: 700; color: #f1f5f9;
      margin: 2px 0;
  }
  .kpi-label {
      font-size: 0.72rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing: .1em;
  }

  /* Section header */
  .dash-section {
      font-size: 1.1rem; font-weight: 600; color: #06b6d4;
      margin: 30px 0 10px; letter-spacing: .04em;
  }

  /* Chart container */
  .chart-card {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(6,182,212,0.15);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 20px;
  }

  /* Detail panel */
  .detail-panel {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(6,182,212,0.3);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 18px;
  }

  .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 10px;
      margin-top: 12px;
  }
  .metric-cell {
      background: rgba(6,182,212,0.08);
      border: 1px solid rgba(6,182,212,0.2);
      border-radius: 10px;
      padding: 10px 14px;
  }
  .metric-cell .label {
      font-size: 0.7rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing:.08em;
  }
  .metric-cell .value {
      font-size: 0.95rem; font-weight: 600; color: #f1f5f9;
      margin-top: 2px;
  }

  /* Img label */
  .img-label {
      text-align: center;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      color: #06b6d4;
      text-transform: uppercase;
      margin-bottom: 6px;
  }

  /* Button */
  div.stButton > button {
      background: linear-gradient(135deg, #0891b2, #0e7490);
      color: #fff;
      border: none;
      border-radius: 10px;
      font-weight: 600;
      padding: 0.55rem 1.4rem;
      transition: transform .15s, box-shadow .15s;
  }
  div.stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(6,182,212,0.4);
  }

  /* Table */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  hr { border-color: rgba(6,182,212,0.15) !important; }

  /* Hide default branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme helper — consistent dark-teal charts
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(6,182,212,0.08)", zerolinecolor="rgba(6,182,212,0.15)"),
    yaxis=dict(gridcolor="rgba(6,182,212,0.08)", zerolinecolor="rgba(6,182,212,0.15)"),
    colorway=[
        "#06b6d4", "#0ea5e9", "#38bdf8", "#7dd3fc",
        "#22d3ee", "#67e8f9", "#a5f3fc", "#0891b2",
    ],
    hoverlabel=dict(bgcolor="#1e293b", font_color="#f1f5f9", bordercolor="#06b6d4"),
)


def _apply_plotly_theme(fig):
    """Apply the project's dark-teal theme to a Plotly figure."""
    fig.update_layout(**_PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------
def _kpi_html(icon: str, value: str, label: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-icon">{icon}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-label">{label}</div>'
        f'</div>'
    )


def _metric_html(label: str, value: str) -> str:
    return (
        f'<div class="metric-cell">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.4rem;font-weight:700;color:#06b6d4;'>"
        "📊 Analytics<br>Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.78rem;color:#94a3b8;'>"
        "Visualise historical water-body detection results."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.7rem;color:#64748b;margin-top:8px;'>"
        "Data source: SQLite · water_bodies.db</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='color:#f1f5f9;margin-bottom:2px;'>📊 Analytics Dashboard</h1>"
    "<p style='color:#94a3b8;margin-top:0;'>"
    "Comprehensive statistics and visualisations of all water-body analyses.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# FETCH DATA (reuse existing database module)
# ═══════════════════════════════════════════════════════════════════════════
try:
    analyses = database.get_all_analyses()
except Exception as exc:
    st.error(f"❌ Failed to load analyses from database: {exc}")
    st.stop()

if not analyses:
    st.info(
        "📭 **No analyses found.**  "
        "Run a **New Analysis** on the main page first, then return here to "
        "see your dashboard populate with charts and statistics."
    )
    st.stop()

df = pd.DataFrame(analyses)

# Parse dates robustly
if "created_at" in df.columns:
    df["date"] = pd.to_datetime(df["created_at"], errors="coerce")
else:
    df["date"] = pd.NaT

# Fill NaN locations for display
df["location_display"] = df["location"].fillna("Unknown")


# ═══════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════
total_analyses   = len(df)
total_water_ha   = df["water_area_ha"].sum() if "water_area_ha" in df.columns else 0
avg_water_pct    = df["water_percentage"].mean() if "water_percentage" in df.columns else 0
max_water_pct    = df["water_percentage"].max() if "water_percentage" in df.columns else 0
total_water_km2  = df["water_area_km2"].sum() if "water_area_km2" in df.columns else 0
unique_locations = df["location"].dropna().nunique()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(_kpi_html("🔬", f"{total_analyses}", "Total Analyses"), unsafe_allow_html=True)
with k2:
    st.markdown(_kpi_html("💧", f"{avg_water_pct:.2f}%", "Avg Water %"), unsafe_allow_html=True)
with k3:
    st.markdown(_kpi_html("📈", f"{max_water_pct:.2f}%", "Peak Water %"), unsafe_allow_html=True)
with k4:
    st.markdown(_kpi_html("🌊", f"{total_water_ha:,.2f}", "Total Water (ha)"), unsafe_allow_html=True)
with k5:
    st.markdown(_kpi_html("🗺️", f"{total_water_km2:,.4f}", "Total Water (km²)"), unsafe_allow_html=True)
with k6:
    st.markdown(_kpi_html("📍", f"{unique_locations}", "Unique Locations"), unsafe_allow_html=True)


st.markdown("")
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# CHARTS — import plotly lazily (ships with Streamlit)
# ═══════════════════════════════════════════════════════════════════════════
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

if _HAS_PLOTLY:

    # ── ROW 1 : Timeline  +  Distribution ───────────────────────────────
    st.markdown('<div class="dash-section">📈 Water Detection Over Time</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if df["date"].notna().any():
            ts = df.dropna(subset=["date"]).sort_values("date")
            fig_time = px.area(
                ts,
                x="date",
                y="water_percentage",
                markers=True,
                labels={"date": "Date", "water_percentage": "Water %"},
                title="Water Coverage % — Timeline",
            )
            fig_time.update_traces(
                line=dict(color="#06b6d4", width=2.5),
                fillcolor="rgba(6,182,212,0.12)",
                marker=dict(size=7, color="#22d3ee"),
            )
            _apply_plotly_theme(fig_time)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.caption("No date information available for timeline chart.")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_dist = px.histogram(
            df,
            x="water_percentage",
            nbins=20,
            title="Water % Distribution",
            labels={"water_percentage": "Water %", "count": "Analyses"},
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_dist.update_traces(marker_line_color="#06b6d4", marker_line_width=1)
        _apply_plotly_theme(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ── ROW 2 : By Location  +  Area Comparison ─────────────────────────
    st.markdown('<div class="dash-section">📍 Analysis by Location</div>', unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2, gap="large")

    with chart_col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        loc_agg = (
            df.groupby("location_display", as_index=False)
            .agg(
                count=("analysis_id", "count"),
                avg_water_pct=("water_percentage", "mean"),
                total_ha=("water_area_ha", "sum"),
            )
            .sort_values("total_ha", ascending=True)
            .tail(10)
        )
        fig_loc = px.bar(
            loc_agg,
            y="location_display",
            x="total_ha",
            orientation="h",
            title="Top Locations by Total Water Area (ha)",
            labels={"location_display": "Location", "total_ha": "Total Water (ha)"},
            color="avg_water_pct",
            color_continuous_scale=["#0c4a6e", "#06b6d4", "#67e8f9"],
        )
        _apply_plotly_theme(fig_loc)
        fig_loc.update_layout(coloraxis_colorbar=dict(title="Avg %", tickfont=dict(size=10)))
        st.plotly_chart(fig_loc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_area = px.bar(
            df.sort_values("water_area_ha", ascending=False).head(15),
            x="analysis_id",
            y="water_area_ha",
            title="Water Area per Analysis (ha) — Top 15",
            labels={"analysis_id": "Analysis", "water_area_ha": "Water Area (ha)"},
            color="water_percentage",
            color_continuous_scale=["#0c4a6e", "#06b6d4", "#22d3ee"],
        )
        _apply_plotly_theme(fig_area)
        fig_area.update_layout(
            xaxis_tickangle=-45,
            coloraxis_colorbar=dict(title="Water %", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_area, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ── ROW 3 : Water Pixels scatter  +  Pie ────────────────────────────
    st.markdown('<div class="dash-section">🔍 Detailed Metrics</div>', unsafe_allow_html=True)

    chart_col5, chart_col6 = st.columns(2, gap="large")

    with chart_col5:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df,
            x="water_pixels",
            y="water_percentage",
            size="water_area_ha",
            color="location_display",
            hover_data=["analysis_id", "water_area_ha"],
            title="Water Pixels vs Water % (bubble = area)",
            labels={
                "water_pixels": "Water Pixels",
                "water_percentage": "Water %",
                "location_display": "Location",
            },
        )
        _apply_plotly_theme(fig_scatter)
        fig_scatter.update_layout(showlegend=True, legend=dict(
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col6:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Pie: water vs non-water across ALL analyses (aggregate pixels)
        total_water_px = int(df["water_pixels"].sum()) if "water_pixels" in df.columns else 0
        # Rough total pixels: each analysis is at original resolution; approximate
        # using water_percentage to back-calculate total
        if avg_water_pct > 0:
            approx_total_px = int(total_water_px / (avg_water_pct / 100))
        else:
            approx_total_px = total_water_px
        non_water_px = max(approx_total_px - total_water_px, 0)

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Water", "Non-Water"],
            values=[total_water_px, non_water_px],
            hole=0.55,
            marker=dict(colors=["#06b6d4", "#1e293b"], line=dict(color="#0d1117", width=2)),
            textfont=dict(color="#f1f5f9"),
        )])
        fig_pie.update_layout(title="Aggregate Water vs Non-Water Pixels")
        _apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning(
        "⚠️ **Plotly is not installed.** Install it with `pip install plotly` "
        "to enable interactive charts.  Falling back to summary table."
    )

# ═══════════════════════════════════════════════════════════════════════════
# DATA TABLE — full history
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="dash-section">🗂️ Complete Analysis History</div>', unsafe_allow_html=True)

display_cols = [
    c for c in [
        "analysis_id", "image_name", "location",
        "water_percentage", "water_area_ha", "water_area_km2",
        "water_pixels", "created_at",
    ] if c in df.columns
]

rename_map = {
    "analysis_id":      "Analysis ID",
    "image_name":       "Image",
    "location":         "Location",
    "water_percentage": "Water %",
    "water_area_ha":    "Water Area (ha)",
    "water_area_km2":   "Water Area (km²)",
    "water_pixels":     "Water Pixels",
    "created_at":       "Date / Time",
}

st.dataframe(
    df[display_cols].rename(columns=rename_map),
    use_container_width=True,
    hide_index=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# DETAIL VIEWER — pick one analysis to inspect with images
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="dash-section">🔎 Analysis Detail Viewer</div>', unsafe_allow_html=True)

analysis_ids = df["analysis_id"].tolist()
selected_id = st.selectbox(
    "Select an analysis to inspect:",
    analysis_ids,
    key="dashboard_detail_select",
)

if selected_id:
    try:
        record = database.get_analysis_by_id(selected_id)
    except Exception as exc:
        record = None
        st.error(f"❌ Database error: {exc}")

    if record is None:
        st.warning(f"Analysis '{selected_id}' not found in database.")
    else:
        st.markdown('<div class="detail-panel">', unsafe_allow_html=True)

        # ── Metrics row ──────────────────────────────────────────────
        metrics_html = '<div class="metric-grid">'
        metrics_html += _metric_html("Analysis ID",     record.get("analysis_id", "—"))
        metrics_html += _metric_html("Image",           record.get("image_name") or "—")
        metrics_html += _metric_html("Location",        record.get("location") or "—")
        metrics_html += _metric_html("Latitude",        str(record.get("latitude") or "—"))
        metrics_html += _metric_html("Longitude",       str(record.get("longitude") or "—"))
        metrics_html += _metric_html("Water Pixels",    f"{record.get('water_pixels', 0):,}")
        metrics_html += _metric_html("Water %",         f"{record.get('water_percentage', 0):.2f} %")
        metrics_html += _metric_html("Water Area (m²)", f"{record.get('water_area_m2', 0):,.2f}")
        metrics_html += _metric_html("Water Area (ha)", f"{record.get('water_area_ha', 0):,.4f}")
        metrics_html += _metric_html("Water Area (km²)",f"{record.get('water_area_km2', 0):,.6f}")
        metrics_html += _metric_html("Date / Time",     record.get("created_at", "—"))
        metrics_html += '</div>'
        st.markdown(metrics_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Load and show stored images ──────────────────────────────
        import cv2

        orig_rel = record.get("original_image_path", "")
        mask_rel = record.get("predicted_mask_path", "")
        orig_abs = os.path.join(config.PROJECT_DIR, orig_rel.replace("/", os.sep))
        mask_abs = os.path.join(config.PROJECT_DIR, mask_rel.replace("/", os.sep))

        has_orig = os.path.exists(orig_abs)
        has_mask = os.path.exists(mask_abs)

        if has_orig or has_mask:
            img_col1, img_col2 = st.columns(2, gap="large")

            if has_orig:
                orig_bgr = cv2.imread(orig_abs)
                if orig_bgr is not None:
                    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    with img_col1:
                        st.markdown(
                            '<div class="img-label">🛰️ Original Image</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(orig_rgb, use_container_width=True)
                else:
                    with img_col1:
                        st.caption("⚠️ Could not decode original image.")
            else:
                with img_col1:
                    st.caption("Original image file not found on disk.")

            if has_mask and has_orig and orig_bgr is not None:
                mask_gray = cv2.imread(mask_abs, cv2.IMREAD_GRAYSCALE)
                if mask_gray is not None:
                    mask_bin = (mask_gray > 127).astype(np.uint8)
                    mask_bin_r = cv2.resize(
                        mask_bin,
                        (orig_bgr.shape[1], orig_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_colour = np.zeros_like(orig_bgr)
                    mask_colour[mask_bin_r == 1] = [255, 160, 0]
                    overlay = cv2.addWeighted(orig_bgr, 0.55, mask_colour, 0.45, 0)
                    mask_disp = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    with img_col2:
                        st.markdown(
                            '<div class="img-label">🔵 Predicted Mask</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(mask_disp, use_container_width=True)
                else:
                    with img_col2:
                        st.caption("⚠️ Could not decode mask image.")
            elif has_mask:
                with img_col2:
                    st.caption("Mask available but original image missing — cannot overlay.")
            else:
                with img_col2:
                    st.caption("Mask image file not found on disk.")
        else:
            st.caption(
                "📁 Image files not found on disk.  They may have been moved or deleted."
            )


# ═══════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="dash-section">💾 Export Data</div>', unsafe_allow_html=True)

export_df = df[display_cols].rename(columns=rename_map)
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️  Download All Analyses as CSV",
    data=csv_bytes,
    file_name=f"water_body_analyses_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    key="dashboard_csv_download",
)

st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.75rem;margin-top:40px;'>"
    "U-Net · 512×512 · Sentinel-2 · Analytics Dashboard"
    "</p>",
    unsafe_allow_html=True,
)
