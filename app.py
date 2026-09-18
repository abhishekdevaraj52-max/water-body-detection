"""
app.py — Water Body Analysis (Streamlit)

Pages
-----
  1. New Analysis  — upload a Sentinel-2 image, run U-Net prediction,
                     save results, display them.
  2. Analysis History — browse past analyses stored in SQLite, view any
                        saved result without re-running the model.

Architecture
------------
  • Images stored as files:  outputs/original/analysis_NNNN.png
                              outputs/masks/analysis_NNNN_mask.png
  • Metadata in SQLite:       database/water_bodies.db  (relative paths)
  • Model never retrained or replaced — inference only.
"""

import os
import sys
import datetime
import time
import traceback

import cv2
import numpy as np
import streamlit as st

import config      # IMG_SIZE, WATER_THRESHOLD, MODEL_PATH, path constants
import database    # init_database, insert_analysis, get_all/get_by_id

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Water Body Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ensure DB + directory structure exist on every cold-start
# ---------------------------------------------------------------------------
database.init_database()

# ---------------------------------------------------------------------------
# Custom CSS — professional dark-teal theme
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

  /* Cards */
  .result-card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(6,182,212,0.3);
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 18px;
  }

  /* Column labels */
  .img-label {
      text-align: center;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      color: #06b6d4;
      text-transform: uppercase;
      margin-bottom: 8px;
  }

  /* Metric row */
  .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 14px;
  }
  .metric-cell {
      background: rgba(6,182,212,0.08);
      border: 1px solid rgba(6,182,212,0.2);
      border-radius: 10px;
      padding: 12px 16px;
  }
  .metric-cell .label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing:.08em; }
  .metric-cell .value { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; margin-top: 2px; }

  /* Divider */
  hr { border-color: rgba(6,182,212,0.15) !important; }

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

  /* Table in history */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* Hide default Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached — loaded ONCE per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading U-Net model…")
def load_unet_model():
    """Load the U-Net model once and warm-up the TF graph so the first
    real inference call is fast rather than spending time on XLA compilation."""
    if not os.path.exists(config.MODEL_PATH):
        return None, f"Model not found at: {config.MODEL_PATH}"
    try:
        t0 = time.perf_counter()
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        m = load_model(config.MODEL_PATH)
        load_secs = time.perf_counter() - t0
        print(f"[TIMING] Model load:  {load_secs:.2f}s")
        print(f"[INFO]   Model input shape: {m.input_shape}")

        # ---- Graph warm-up: run one dummy prediction so TF compiles the
        #      computation graph before the user's first real image arrives.
        dummy = np.zeros(
            (1, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32
        )
        _ = m(dummy, training=False)   # triggers tf.function trace + XLA compile
        warmup_secs = time.perf_counter() - t0 - load_secs
        print(f"[TIMING] Graph warm-up: {warmup_secs:.2f}s")
        return m, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def safe_postprocess_mask(prediction: np.ndarray) -> np.ndarray:
    """
    Convert raw model output (2D / 3D / 4D) to a 2D binary uint8 mask
    where 1 = water, 0 = non-water.
    """
    pred = np.squeeze(prediction)       # remove batch + channel dims if present
    if pred.ndim == 3:
        pred = pred[:, :, 0]            # take first channel of (H,W,C)
    elif pred.ndim != 2:
        raise ValueError(f"Unexpected prediction shape after squeeze: {pred.shape}")
    return (pred >= config.WATER_THRESHOLD).astype(np.uint8)


def run_prediction(
    img_bgr: np.ndarray,
    model,
    timing_log: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run U-Net inference on img_bgr (original BGR array).

    Returns (mask_original, mask_display_rgb):
      mask_original  — binary uint8 mask at original image resolution
      mask_display   — colourised overlay (water highlighted) at original res, RGB

    If timing_log is a list it will be populated with (stage, seconds) tuples.
    """
    import tensorflow as tf

    t_total = time.perf_counter()
    original_h, original_w = img_bgr.shape[:2]

    # ── 1. Preprocess ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    img_resized = cv2.resize(img_bgr, (config.IMG_SIZE, config.IMG_SIZE))
    img_norm    = img_resized.astype(np.float32) / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)  # shape (1, 512, 512, 3)
    t_pre = time.perf_counter() - t0
    if timing_log is not None:
        timing_log.append(("Preprocessing", t_pre))
    print(f"[TIMING] Preprocess:  {t_pre*1000:.1f} ms  (input shape {img_input.shape})")

    # ── 2. U-Net inference — call model directly (lower overhead than .predict)
    t0 = time.perf_counter()
    # model() with training=False runs in inference mode without verbose overhead
    prediction_tensor = model(img_input, training=False)
    # Materialise the tensor to a numpy array once
    prediction = prediction_tensor.numpy() if hasattr(prediction_tensor, "numpy") else np.array(prediction_tensor)
    t_inf = time.perf_counter() - t0
    if timing_log is not None:
        timing_log.append(("U-Net inference", t_inf))
    print(f"[TIMING] Inference:   {t_inf*1000:.1f} ms  (output shape {prediction.shape})")

    # ── 3. Postprocess ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    mask_512 = safe_postprocess_mask(prediction)

    # Resize mask back to original image dimensions
    mask_original = cv2.resize(
        mask_512, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    # Build a colourised display image (water = bright cyan/orange highlight)
    mask_colour = np.zeros_like(img_bgr)
    mask_colour[mask_original == 1] = [255, 160, 0]   # BGR: orange-cyan tint
    overlay_bgr  = cv2.addWeighted(img_bgr, 0.55, mask_colour, 0.45, 0)
    mask_display = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    t_post = time.perf_counter() - t0
    if timing_log is not None:
        timing_log.append(("Postprocessing", t_post))
    print(f"[TIMING] Postprocess: {t_post*1000:.1f} ms")

    t_total_elapsed = time.perf_counter() - t_total
    if timing_log is not None:
        timing_log.append(("Total prediction", t_total_elapsed))
    print(f"[TIMING] Total pred:  {t_total_elapsed*1000:.1f} ms")

    return mask_original, mask_display


def calculate_water_stats(mask: np.ndarray) -> dict:
    """Return pixel count + percentage + rough area estimates."""
    total_pixels   = mask.size
    water_pixels   = int(np.sum(mask))
    water_pct      = round((water_pixels / total_pixels) * 100, 4) if total_pixels else 0.0
    # Rough area: assume each pixel ≈ 10 m × 10 m (Sentinel-2 native resolution)
    pixel_area_m2  = 100.0
    water_area_m2  = round(water_pixels * pixel_area_m2, 2)
    water_area_ha  = round(water_area_m2 / 10_000, 4)
    water_area_km2 = round(water_area_m2 / 1_000_000, 6)
    return {
        "water_pixels":   water_pixels,
        "water_pct":      water_pct,
        "water_area_m2":  water_area_m2,
        "water_area_ha":  water_area_ha,
        "water_area_km2": water_area_km2,
    }


def save_images(analysis_id: str, img_bgr: np.ndarray,
                mask_original: np.ndarray) -> tuple[str, str]:
    """
    Persist original + mask images as PNGs.
    Returns (relative_original_path, relative_mask_path).
    """
    orig_filename = f"{analysis_id}.png"
    mask_filename = f"{analysis_id}_mask.png"

    orig_abs = os.path.join(config.ORIG_DIR,  orig_filename)
    mask_abs = os.path.join(config.MASKS_DIR, mask_filename)

    # Original: save in BGR (cv2 native)
    cv2.imwrite(orig_abs, img_bgr)

    # Mask: save as white/black binary PNG
    mask_png = (mask_original * 255).astype(np.uint8)
    cv2.imwrite(mask_abs, mask_png)

    # Return relative paths (forward slashes for portability)
    rel_orig = f"outputs/original/{orig_filename}"
    rel_mask = f"outputs/masks/{mask_filename}"
    return rel_orig, rel_mask


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _metric_html(label: str, value: str) -> str:
    return (
        f'<div class="metric-cell">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{value}</div>'
        f'</div>'
    )


def display_result(
    orig_img_rgb: np.ndarray,
    mask_display: np.ndarray,
    record: dict,
) -> None:
    """Render the two-column image view + metrics for a single analysis."""

    # ---- Header ---------------------------------------------------------
    st.markdown("---")
    location_str = record.get("location") or "—"
    st.markdown(
        f"<h2 style='text-align:center;color:#06b6d4;margin-bottom:4px;'>🌊 Water Body Analysis</h2>"
        f"<p style='text-align:center;color:#94a3b8;font-size:0.95rem;'>📍 {location_str}</p>",
        unsafe_allow_html=True,
    )

    # ---- Images ---------------------------------------------------------
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="img-label">🛰️ Sentinel-2 (Original)</div>', unsafe_allow_html=True)
        st.image(orig_img_rgb, use_container_width=True)
    with col2:
        st.markdown('<div class="img-label">🔵 Predicted Mask (Water)</div>', unsafe_allow_html=True)
        st.image(mask_display, use_container_width=True)

    # ---- Metrics --------------------------------------------------------
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    metrics_html = '<div class="metric-grid">'
    metrics_html += _metric_html("Analysis ID",     record.get("analysis_id", "—"))
    metrics_html += _metric_html("Location",        record.get("location") or "—")
    metrics_html += _metric_html("Latitude",        str(record.get("latitude") or "—"))
    metrics_html += _metric_html("Longitude",       str(record.get("longitude") or "—"))
    metrics_html += _metric_html("Water Pixels",    f"{record.get('water_pixels', 0):,}")
    metrics_html += _metric_html("Water %",         f"{record.get('water_percentage', 0):.2f} %")
    metrics_html += _metric_html("Water Area (m²)", f"{record.get('water_area_m2', 0):,.2f} m²")
    metrics_html += _metric_html("Water Area (ha)", f"{record.get('water_area_ha', 0):,.4f} ha")
    metrics_html += _metric_html("Water Area (km²)",f"{record.get('water_area_km2', 0):,.6f} km²")
    metrics_html += _metric_html("Prediction Time", record.get("created_at", "—"))
    metrics_html += '</div>'
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def load_result_images(record: dict) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """
    Load stored original + mask images from disk using relative paths in DB.
    Returns (orig_rgb, mask_display, error_message).
    """
    project_dir = config.PROJECT_DIR
    orig_path   = os.path.join(project_dir, record["original_image_path"].replace("/", os.sep))
    mask_path   = os.path.join(project_dir, record["predicted_mask_path"].replace("/", os.sep))

    if not os.path.exists(orig_path):
        return None, None, f"Original image not found: {orig_path}"
    if not os.path.exists(mask_path):
        return None, None, f"Mask image not found: {mask_path}"

    orig_bgr  = cv2.imread(orig_path)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if orig_bgr is None:
        return None, None, f"Failed to decode original image: {orig_path}"
    if mask_gray is None:
        return None, None, f"Failed to decode mask image: {mask_path}"

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # Reconstruct colourised mask display
    mask_bin   = (mask_gray > 127).astype(np.uint8)
    mask_color = np.zeros_like(orig_bgr)
    h, w       = min(orig_bgr.shape[0], mask_bin.shape[0]), min(orig_bgr.shape[1], mask_bin.shape[1])
    mask_bin_r = cv2.resize(mask_bin, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_color[mask_bin_r == 1] = [255, 160, 0]
    overlay    = cv2.addWeighted(orig_bgr, 0.55, mask_color, 0.45, 0)
    mask_disp  = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return orig_rgb, mask_disp, None


# ===========================================================================
# SIDEBAR — navigation
# ===========================================================================

with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.4rem;font-weight:700;color:#06b6d4;'>🌊 Water Body<br>Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 New Analysis", "📂 Analysis History"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#64748b;'>U-Net · 512×512 · Sentinel-2</p>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — New Analysis
# ===========================================================================

if page == "🔬 New Analysis":
    st.markdown(
        "<h1 style='color:#f1f5f9;margin-bottom:4px;'>🔬 New Water Body Analysis</h1>"
        "<p style='color:#94a3b8;'>Upload a Sentinel-2 satellite image to detect and measure water bodies.</p>",
        unsafe_allow_html=True,
    )

    # ---- Model status ---------------------------------------------------
    model, model_err = load_unet_model()
    if model_err:
        st.error(f"⚠️ Model error: {model_err}")
        st.stop()
    else:
        st.success(f"✅ U-Net loaded — input shape: `{model.input_shape}`")

    st.markdown("---")

    # ---- Location inputs ------------------------------------------------
    with st.expander("📍 Location (optional)", expanded=False):
        loc_name = st.text_input("Location name", placeholder="e.g. Ganzourgou, Burkina Faso")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat_input = st.text_input("Latitude",  placeholder="e.g. 12.3456")
        with col_lon:
            lon_input = st.text_input("Longitude", placeholder="e.g. -1.2345")

    def _parse_coord(val: str):
        try:
            return float(val.strip()) if val and val.strip() else None
        except ValueError:
            return None

    lat  = _parse_coord(lat_input)
    lon  = _parse_coord(lon_input)

    # ---- File uploader --------------------------------------------------
    uploaded = st.file_uploader(
        "Upload Sentinel-2 image", type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Supports JPEG, PNG, and GeoTIFF (single-band or RGB).",
    )

    if uploaded is not None:
        # Preview the upload
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Could not decode the uploaded file. Please upload a valid image.")
            st.stop()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Uploaded: {uploaded.name}  ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)",
                 use_container_width=True)

        st.markdown("---")

        # ---- Two action buttons side-by-side ----------------------------
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            detect_clicked = st.button(
                "🔍 Detect Water Bodies",
                use_container_width=True,
                key="detect_btn",
            )
        with btn_col2:
            clear_clicked = st.button(
                "🗑️ Clear",
                use_container_width=True,
                key="clear_btn",
            )

        if clear_clicked:
            # Re-run the page which resets the uploader
            st.rerun()

        if detect_clicked:
            timing_log: list = []
            analysis_id = None
            mask_original = None
            mask_display  = None
            stats         = None

            # ── Stage 1: inference ─────────────────────────────────────────
            status_placeholder = st.empty()
            status_placeholder.info("⚙️ Detecting water bodies…")
            try:
                t_wall = time.perf_counter()
                mask_original, mask_display = run_prediction(
                    img_bgr, model, timing_log=timing_log
                )
                elapsed_inf = time.perf_counter() - t_wall
            except Exception as exc:
                status_placeholder.empty()
                st.error(f"❌ U-Net inference failed: {exc}")
                with st.expander("🔍 Full traceback"):
                    st.code(traceback.format_exc())
                st.stop()

            # ── Stage 2: stats ─────────────────────────────────────────────
            stats = calculate_water_stats(mask_original)

            # ── Stage 3: generate analysis ID ──────────────────────────────
            analysis_id = database.generate_analysis_id()

            # ── Clear status, show result immediately ───────────────────────
            status_placeholder.empty()
            st.success(
                f"✅ Analysis complete — ID: **{analysis_id}** "
                f"({elapsed_inf*1000:.0f} ms)"
            )
            display_result(img_rgb, mask_display, {
                "analysis_id":      analysis_id,
                "image_name":       uploaded.name,
                "location":         loc_name or None,
                "latitude":         lat,
                "longitude":        lon,
                "water_pixels":     stats["water_pixels"],
                "water_percentage": stats["water_pct"],
                "water_area_m2":    stats["water_area_m2"],
                "water_area_ha":    stats["water_area_ha"],
                "water_area_km2":   stats["water_area_km2"],
                "created_at":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            # ── Timing details (expandable) ─────────────────────────────────
            with st.expander("⏱️ Timing breakdown", expanded=False):
                for stage, secs in timing_log:
                    st.write(f"**{stage}:** {secs*1000:.1f} ms")

            # ── Stage 4: disk I/O + DB (after result is visible) ──────────
            try:
                rel_orig, rel_mask = save_images(analysis_id, img_bgr, mask_original)
                database.insert_analysis(
                    analysis_id         = analysis_id,
                    image_name          = uploaded.name,
                    original_image_path = rel_orig,
                    predicted_mask_path = rel_mask,
                    location            = loc_name or None,
                    latitude            = lat,
                    longitude           = lon,
                    water_area_m2       = stats["water_area_m2"],
                    water_area_ha       = stats["water_area_ha"],
                    water_area_km2      = stats["water_area_km2"],
                    water_pixels        = stats["water_pixels"],
                    water_percentage    = stats["water_pct"],
                )
                st.markdown(
                    f"<p style='color:#64748b;font-size:0.8rem;'>💾 Original saved to "
                    f"<code>{rel_orig}</code><br>"
                    f"💾 Mask saved to <code>{rel_mask}</code></p>",
                    unsafe_allow_html=True,
                )
            except Exception as save_exc:
                st.warning(f"⚠️ Result displayed but could not save to disk/DB: {save_exc}")

    else:
        st.markdown(
            "<div style='text-align:center;padding:60px 0;color:#475569;'>"
            "<div style='font-size:3rem;'>🛰️</div>"
            "<p>Upload a Sentinel-2 image above to begin analysis.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 2 — Analysis History
# ===========================================================================

elif page == "📂 Analysis History":
    st.markdown(
        "<h1 style='color:#f1f5f9;margin-bottom:4px;'>📂 Analysis History</h1>"
        "<p style='color:#94a3b8;'>Browse and replay previous water body analyses.</p>",
        unsafe_allow_html=True,
    )

    analyses = database.get_all_analyses()

    if not analyses:
        st.info("No analyses found. Run a **New Analysis** first.")
        st.stop()

    # ---- Summary table --------------------------------------------------
    import pandas as pd

    df = pd.DataFrame(analyses)[[
        "analysis_id", "image_name", "location",
        "water_area_ha", "water_percentage", "created_at",
    ]].rename(columns={
        "analysis_id":    "Analysis ID",
        "image_name":     "Image",
        "location":       "Location",
        "water_area_ha":  "Water Area (ha)",
        "water_percentage": "Water %",
        "created_at":     "Date / Time",
    })

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # ---- Detail selector ------------------------------------------------
    analysis_ids = [a["analysis_id"] for a in analyses]
    selected_id  = st.selectbox("Select an analysis to view in detail:", analysis_ids)

    if selected_id:
        record = database.get_analysis_by_id(selected_id)
        if record is None:
            st.error(f"Analysis '{selected_id}' not found in database.")
            st.stop()

        orig_rgb, mask_disp, err = load_result_images(record)
        if err:
            st.error(f"❌ {err}")
            st.stop()

        display_result(orig_rgb, mask_disp, record)
