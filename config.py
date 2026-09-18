import os

# ---------------------------------------------------------------------------
# Model / inference settings
# ---------------------------------------------------------------------------

# Inference image size – the model was trained on 512×512 images.
IMG_SIZE = 512

# Water detection probability threshold (default 0.5).
WATER_THRESHOLD = float(os.getenv('WATER_THRESHOLD', '0.5'))

# ---------------------------------------------------------------------------
# Paths – can be overridden via environment variables if needed.
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.getenv('MODEL_PATH', os.path.join(PROJECT_DIR, 'models', 'water_unet.h5'))

# Legacy Flask paths (keep for server.py compatibility)
TMP_DIR     = os.path.join(PROJECT_DIR, 'tmp')
EXPORTS_DIR = os.path.join(PROJECT_DIR, 'exports')

# ---------------------------------------------------------------------------
# Streamlit / database paths
# ---------------------------------------------------------------------------
DB_DIR      = os.path.join(PROJECT_DIR, 'database')
DB_PATH     = os.path.join(DB_DIR, 'water_bodies.db')

OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')
ORIG_DIR    = os.path.join(OUTPUTS_DIR, 'original')
MASKS_DIR   = os.path.join(OUTPUTS_DIR, 'masks')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# ---------------------------------------------------------------------------
# Ensure all output directories exist on import
# ---------------------------------------------------------------------------
for _d in [TMP_DIR, EXPORTS_DIR, DB_DIR, ORIG_DIR, MASKS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
