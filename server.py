import os
import config
import cv2
import numpy as np
import base64
import sys
import platform
import time
import subprocess
import threading
import json
import requests
import uuid
import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
import rasterio
import geopandas as gpd
import shapely.geometry as sgeom
import rasterio.features
import rasterio.warp
import rasterio.transform
# pyrefly: ignore [missing-import]
import rasterio.crs

# Earth Engine import (optional)
try:
    import ee
except Exception:
    ee = None

# Google Auth for ADC (Application Default Credentials)
try:
    import google.auth
    import google.auth.transport.requests
except Exception:
    google = None

GEE_PROJECT = os.getenv('GEE_PROJECT', 'water-detection-abhi2020')
_ee_initialized = False

def init_ee():
    """Initialize Earth Engine using ADC credentials. Called lazily."""
    global _ee_initialized
    if _ee_initialized:
        return True
    if ee is None:
        raise RuntimeError('earthengine-api not installed.')
    try:
        ee.Initialize(project=GEE_PROJECT)
        _ee_initialized = True
        print(f'[OK] Earth Engine initialized with project: {GEE_PROJECT}')
        return True
    except Exception as ex:
        raise RuntimeError(f'Earth Engine init failed: {ex}')

app = Flask(__name__)

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(PROJECT_DIR, 'models', 'water_unet.h5'))
WATER_THRESHOLD = float(os.getenv('WATER_THRESHOLD', '0.5'))
TMP_DIR = os.path.join(PROJECT_DIR, 'tmp')
EXPORTS_DIR = os.path.join(PROJECT_DIR, 'exports')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
model = None
server_start_time = time.time()
prediction_count = 0

# Command task execution memory
command_tasks = {}
task_lock = threading.Lock()

def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading U-Net model from {MODEL_PATH}...")
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH)
            print(f"[INFO] Model input shape: {model.input_shape}")
            print("[OK] U-Net model loaded successfully into memory!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
    return model



def get_model_info():
    exists = os.path.exists(MODEL_PATH)
    size_mb = round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2) if exists else 0
    m = get_model()
    params = m.count_params() if m is not None else 0
    return {
        'exists': exists,
        'path': MODEL_PATH,
        'size_mb': size_mb,
        'parameters': params,
        'loaded': m is not None
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    m_info = get_model_info()
    uptime_sec = int(time.time() - server_start_time)
    return jsonify({
        'status': 'online',
        'server_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'uptime': f"{uptime_sec // 60}m {uptime_sec % 60}s",
        'model_loaded': m_info['loaded'],
        'model_exists': m_info['exists'],
        'model_size_mb': m_info['size_mb'],
        'model_params': m_info['parameters'],
        'predictions_served': prediction_count,
        'python_version': platform.python_version(),
        'os': f"{platform.system()} {platform.release()}"
    })

@app.route('/api/commands', methods=['GET'])
def list_commands():
    commands = [
        {
            'id': 'system_info',
            'name': '📊 System & Model Health Check',
            'description': 'Inspect model status, RAM parameters, hardware specs and server uptime.',
            'category': 'Diagnostics',
            'badge': 'Fast',
            'shortcut': 'Mod+1'
        },
        {
            'id': 'reload_model',
            'name': '🔄 Reload Model into Memory',
            'description': 'Force re-load the U-Net model weights from models/water_unet.h5.',
            'category': 'Model',
            'badge': 'Action',
            'shortcut': 'Mod+2'
        },
        {
            'id': 'check_deps',
            'name': '📦 Verify Dependencies',
            'description': 'Check status of TensorFlow, OpenCV, Flask, NumPy installation.',
            'category': 'Diagnostics',
            'badge': 'Fast',
            'shortcut': 'Mod+3'
        },
        {
            'id': 'train_fast',
            'name': '⚡ Run Quick Model Test (train_fast.py)',
            'description': 'Run rapid training benchmark test using 40 sample satellite images.',
            'category': 'Training',
            'badge': 'Background',
            'shortcut': 'Mod+4'
        },
        {
            'id': 'train_all',
            'name': '🧠 Train Full Model (train_all.py)',
            'description': 'Train U-Net on all 2841 satellite images (runs asynchronously).',
            'category': 'Training',
            'badge': 'Heavy',
            'shortcut': 'Mod+5'
        }
    ]
    return jsonify({'commands': commands})

def execute_background_command(task_id, script_name):
    script_path = os.path.join(PROJECT_DIR, script_name)
    if not os.path.exists(script_path):
        with task_lock:
            command_tasks[task_id]['status'] = 'failed'
            command_tasks[task_id]['output'].append(f"[ERROR] Script {script_name} not found.")
        return

    with task_lock:
        command_tasks[task_id]['status'] = 'running'
        command_tasks[task_id]['output'].append(f"[START] Executing python {script_name}...")

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_DIR,
            bufsize=1
        )
        for line in proc.stdout:
            with task_lock:
                command_tasks[task_id]['output'].append(line.rstrip())
        proc.wait()
        
        with task_lock:
            if proc.returncode == 0:
                command_tasks[task_id]['status'] = 'completed'
                command_tasks[task_id]['output'].append(f"[SUCCESS] {script_name} completed successfully.")
            else:
                command_tasks[task_id]['status'] = 'failed'
                command_tasks[task_id]['output'].append(f"[ERROR] {script_name} failed with return code {proc.returncode}.")
    except Exception as e:
        with task_lock:
            command_tasks[task_id]['status'] = 'failed'
            command_tasks[task_id]['output'].append(f"[EXCEPTION] {str(e)}")

@app.route('/api/run-command', methods=['POST'])
def run_command_endpoint():
    data = request.get_json() or {}
    cmd_id = data.get('command')
    
    if not cmd_id:
        return jsonify({'error': 'No command ID specified.'}), 400

    task_id = f"{cmd_id}_{int(time.time() * 1000)}"

    if cmd_id == 'system_info':
        info = get_model_info()
        uptime_sec = int(time.time() - server_start_time)
        output = [
            "============================================================",
            "  SYSTEM & MODEL DIAGNOSTIC REPORT",
            "============================================================",
            f"Server Status     : ONLINE",
            f"Server Uptime     : {uptime_sec // 60} minutes {uptime_sec % 60} seconds",
            f"Python Version    : {platform.python_version()}",
            f"OS Platform       : {platform.system()} {platform.release()} ({platform.machine()})",
            f"Project Directory : {PROJECT_DIR}",
            "------------------------------------------------------------",
            f"Model File Path   : {info['path']}",
            f"Model Exists      : {'YES' if info['exists'] else 'NO'}",
            f"Model File Size   : {info['size_mb']} MB",
            f"Model Loaded      : {'YES' if info['loaded'] else 'NO'}",
            f"Model Parameters  : {info['parameters']:,} parameters",
            f"Total Predictions : {prediction_count}",
            "============================================================"
        ]
        with task_lock:
            command_tasks[task_id] = {'status': 'completed', 'output': output, 'command': cmd_id}
        return jsonify({'task_id': task_id, 'status': 'completed', 'output': output})

    elif cmd_id == 'reload_model':
        global model
        output = ["Re-loading U-Net model from disk..."]
        try:
            model = None
            m = get_model()
            if m is not None:
                output.append(f"[SUCCESS] Model reloaded successfully! Total parameters: {m.count_params():,}")
                status_str = 'completed'
            else:
                output.append("[WARNING] Model file missing or failed to load.")
                status_str = 'failed'
        except Exception as e:
            output.append(f"[ERROR] Reload failed: {str(e)}")
            status_str = 'failed'
        
        with task_lock:
            command_tasks[task_id] = {'status': status_str, 'output': output, 'command': cmd_id}
        return jsonify({'task_id': task_id, 'status': status_str, 'output': output})

    elif cmd_id == 'check_deps':
        deps = ['flask', 'tensorflow', 'cv2', 'numpy', 'PIL']
        output = ["Checking core Python package dependencies..."]
        all_ok = True
        for dep in deps:
            try:
                mod = __import__(dep)
                ver = getattr(mod, '__version__', 'Installed')
                output.append(f"  [OK] {dep:12s} : v{ver}")
            except ImportError:
                output.append(f"  [MISSING] {dep:12s} : NOT INSTALLED")
                all_ok = False
        output.append("All required dependencies are satisfied!" if all_ok else "Some dependencies are missing. Run: pip install -r requirements.txt")
        
        with task_lock:
            command_tasks[task_id] = {'status': 'completed' if all_ok else 'failed', 'output': output, 'command': cmd_id}
        return jsonify({'task_id': task_id, 'status': 'completed' if all_ok else 'failed', 'output': output})

    elif cmd_id in ['train_fast', 'train_all']:
        script_map = {'train_fast': 'train_fast.py', 'train_all': 'train_all.py'}
        script_name = script_map[cmd_id]
        
        with task_lock:
            command_tasks[task_id] = {'status': 'pending', 'output': [], 'command': cmd_id}
        
        thread = threading.Thread(target=execute_background_command, args=(task_id, script_name), daemon=True)
        thread.start()
        return jsonify({'task_id': task_id, 'status': 'running', 'output': [f"Launched {script_name} in background..."]})

    else:
        return jsonify({'error': f"Unknown command: {cmd_id}"}), 400

@app.route('/api/command-logs', methods=['GET'])
def command_logs():
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'Missing task_id parameter.'}), 400
    
    with task_lock:
        task = command_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task ID not found.'}), 404
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'output': task['output']
        })

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_count
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Model not found at models/water_unet.h5. Please train the model first.'}), 503
    try:
        # Read uploaded image — preserve original dimensions
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Failed to decode image.'}), 400

        original_h, original_w = img_bgr.shape[:2]

        # Resize a COPY to 512×512 for U-Net inference
        img_resized = cv2.resize(img_bgr, (config.IMG_SIZE, config.IMG_SIZE))

        # Debug logging
        print("========================================")
        print("U-NET INFERENCE (/predict)")
        print("========================================")
        print(f"Model input shape: {current_model.input_shape}")
        print(f"Original image shape: {img_bgr.shape}")
        print(f"Inference image shape: {img_resized.shape}")

        # Preprocess and predict
        img_input = preprocess_image(img_resized)
        print(f"Input tensor shape: {img_input.shape}")

        prediction = current_model.predict(img_input, verbose=0)
        print(f"Prediction shape: {prediction.shape}")

        mask_512 = (prediction[0, :, :, 0] >= config.WATER_THRESHOLD).astype(np.uint8)

        # Resize mask back to original image dimensions
        mask_original = cv2.resize(
            mask_512,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        print(f"Final mask shape: {mask_original.shape}")
        print(f"Water threshold: {config.WATER_THRESHOLD}")
        print("========================================")

        mask_bw = (mask_original * 255).astype(np.uint8)

        # Create overlay on original-size image
        mask_bgr = np.zeros_like(img_bgr)
        mask_bgr[mask_original == 1] = [255, 160, 0]
        overlay = cv2.addWeighted(img_bgr, 0.6, mask_bgr, 0.4, 0)

        # Encode images to base64
        _, buffer_orig = cv2.imencode('.jpg', img_bgr)
        _, buffer_mask = cv2.imencode('.jpg', mask_bw)
        _, buffer_overlay = cv2.imencode('.jpg', overlay)
        b64_orig = base64.b64encode(buffer_orig).decode('utf-8')
        b64_mask = base64.b64encode(buffer_mask).decode('utf-8')
        b64_overlay = base64.b64encode(buffer_overlay).decode('utf-8')
        total_pixels = mask_original.size
        water_pixels = int(np.sum(mask_original))
        water_percent = round((water_pixels / total_pixels) * 100, 2)
        prediction_count += 1
        return jsonify({
            'original': f"data:image/jpeg;base64,{b64_orig}",
            'mask': f"data:image/jpeg;base64,{b64_mask}",
            'overlay': f"data:image/jpeg;base64,{b64_overlay}",
            'water_pixels': water_pixels,
            'water_percent': water_percent
        })

    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500

# Existing routes and functions remain unchanged up to line 326
# Insert new helper functions and route after the existing /predict route

# Helper: Reverse geocode using Nominatim
def reverse_geocode(lat, lon):
    try:
        resp = requests.get('https://nominatim.openstreetmap.org/reverse', params={
            'format': 'json',
            'lat': lat,
            'lon': lon,
            'zoom': 10,
            'addressdetails': 1
        }, headers={'User-Agent': 'water-body-app'})
        if resp.status_code != 200:
            return None, f"Nominatim error: {resp.status_code}"
        data = resp.json()
        name = data.get('display_name', '')
        return name, None
    except Exception as e:
        return None, str(e)

# Helper: Fetch Sentinel-2 image from GEE using pixel download API
def fetch_sentinel_image(lat, lon, uid):
    """Download a 512x512 Sentinel-2 RGB GeoTIFF using EE pixel API."""
    init_ee()  # ensure authenticated
    point = ee.Geometry.Point(lon, lat)
    buffer_m = 5000  # 5 km radius
    region = point.buffer(buffer_m).bounds()

    # Try last 180 days with cloud filter
    today = datetime.datetime.utcnow()
    for days_back in [60, 180, 365]:
        start = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        end   = today.strftime('%Y-%m-%d')
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(start, end)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))
        )
        if collection.size().getInfo() > 0:
            break
    else:
        raise RuntimeError('No cloud-free Sentinel-2 image found in the last 365 days for this location.')

    # Median composite, select RGB bands (B4=red, B3=green, B2=blue)
    image = collection.median().select(['B4', 'B3', 'B2'])

    # Use getDownloadURL for a GeoTIFF (512x512 px)
    region_coords = region.getInfo()['coordinates']
    url = image.getDownloadURL({
        'scale': 40,          # ~40m/px → ~250px for 10km box
        'region': region_coords,
        'format': 'GEO_TIFF',
        'bands': ['B4', 'B3', 'B2'],
    })

    export_dir = os.path.join(TMP_DIR, uid)
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'satellite.tif')

    response = requests.get(url, stream=True, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f'GEE download failed ({response.status_code}): {response.text[:200]}')
    with open(export_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with rasterio.open(export_path) as src:
        arr = src.read()               # (3, H, W)  — B4, B3, B2
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        res = src.res
    return arr, transform, crs, bounds, res, export_path

# Helper: Preprocess image for model (512×512 inference)
def preprocess_image(img_array):
    # img_array assumed shape (H,W,3) in BGR (as read by cv2)
    # Resize to 512×512 to match the trained U-Net input
    img_resized = cv2.resize(img_array, (config.IMG_SIZE, config.IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_normalized, axis=0)

# Helper: Postprocess mask and calculate area
def calculate_area(mask, transform, crs):
    # mask shape (512,512)
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import geopandas as gpd
    import shapely.geometry as geometry
    # Write temporary mask raster
    uid = str(uuid.uuid4())
    mask_path = os.path.join(TMP_DIR, uid, 'mask.tif')
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    with rasterio.open(
        mask_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)
    # Reproject to appropriate UTM zone based on centroid
    centroid = rasterio.transform.xy(transform, mask.shape[0]//2, mask.shape[1]//2)
    lon, lat = centroid[0], centroid[1]
    utm_crs = rasterio.crs.CRS.from_string(f"EPSG:{utm_zone_epsg(lat, lon)}")
    dst_path = os.path.join(TMP_DIR, uid, 'mask_utm.tif')
    with rasterio.open(mask_path) as src:
        transform_utm, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': utm_crs,
            'transform': transform_utm,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_utm,
                dst_crs=utm_crs,
                resampling=Resampling.nearest)
    # Compute area in square meters (pixel area sum)
    with rasterio.open(dst_path) as ds:
        pixel_area = abs(ds.transform.a * ds.transform.e)  # meters per pixel squared
        mask_data = ds.read(1)
        water_pixels = np.sum(mask_data > 0)
        area_m2 = water_pixels * pixel_area
    return area_m2, water_pixels, dst_path

# Helper: Convert mask to GeoJSON and Shapefile ZIP
def mask_to_vector(mask_path, uid):
    import geopandas as gpd
    import rasterio.features
    import shapely.geometry as sgeom
    import zipfile, io, os
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        shapes = rasterio.features.shapes(mask, transform=src.transform)
        polys = [sgeom.shape(geom) for geom, value in shapes if value == 1]
        if not polys:
            raise RuntimeError('No water features detected in mask.')
        gdf = gpd.GeoDataFrame({'geometry': polys}, crs=src.crs)
        # Export GeoJSON
        geojson_path = os.path.join(EXPORTS_DIR, uid + '.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
        # Export Shapefile zip
        shp_dir = os.path.join(EXPORTS_DIR, uid + '_shp')
        os.makedirs(shp_dir, exist_ok=True)
        shp_path = os.path.join(shp_dir, uid + '.shp')
        gdf.to_file(shp_path)
        zip_path = os.path.join(EXPORTS_DIR, uid + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(shp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)
        return geojson_path, zip_path

# Helper: Determine UTM EPSG from lat/lon
def utm_zone_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return epsg

# New API endpoint
@app.route('/api/detect-location-water', methods=['POST'])
def detect_location_water():
    data = request.get_json() or {}
    lat = data.get('lat')
    lon = data.get('lon')
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon required'}), 400
    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({'error': 'Invalid lat/lon values'}), 400

    uid = str(uuid.uuid4())

    # Reverse geocode
    place_name, err = reverse_geocode(lat, lon)
    if err:
        place_name = f'{lat:.4f}, {lon:.4f}'

    # Fetch satellite image from GEE
    try:
        img_array, transform, crs, bounds, res, sat_path = fetch_sentinel_image(lat, lon, uid)
    except Exception as e:
        return jsonify({'error': f'Satellite retrieval failed: {str(e)}'}), 500

    # Preserve original raster dimensions for GIS processing
    original_h, original_w = img_array.shape[:2]

    # Normalize Sentinel-2 reflectance (0–10000) → 0–255
    img_vis = img_array.astype(np.float32)
    p2, p98 = np.percentile(img_vis, 2), np.percentile(img_vis, 98)
    if p98 > p2:
        img_vis = np.clip((img_vis - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    else:
        img_vis = np.clip(img_vis / 100, 0, 255).astype(np.uint8)

    # Convert to BGR for cv2 processing (img_array is RGB: B4=R, B3=G, B2=B)
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Resize a COPY to 512×512 for U-Net inference — keep original intact
    img_resized = cv2.resize(img_bgr, (config.IMG_SIZE, config.IMG_SIZE))

    # Load model
    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Model not found or failed to load. Train the model first.'}), 503

    # Preprocess and predict
    try:
        print("========================================")
        print("U-NET INFERENCE (/api/detect-location-water)")
        print("========================================")
        print(f"Model input shape: {current_model.input_shape}")
        print(f"Original image shape: ({original_h}, {original_w}, {img_array.shape[2]})")
        print(f"Inference image shape: {img_resized.shape}")

        img_input = preprocess_image(img_resized)
        print(f"Input tensor shape: {img_input.shape}")

        prediction = current_model.predict(img_input, verbose=0)
        print(f"Prediction shape: {prediction.shape}")

        mask_512 = (prediction[0, :, :, 0] >= WATER_THRESHOLD).astype(np.uint8)

        # Resize mask back to ORIGINAL raster dimensions
        mask_original = cv2.resize(
            mask_512,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        print(f"Final mask shape: {mask_original.shape}")
        print(f"Water threshold: {WATER_THRESHOLD}")
        print("========================================")
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Build colorized overlay on the 512×512 image for display
    mask_display = cv2.resize(
        mask_original,
        (config.IMG_SIZE, config.IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    mask_bgr = np.zeros_like(img_resized)
    mask_bgr[mask_display == 1] = [255, 160, 0]   # cyan-ish in BGR
    overlay = cv2.addWeighted(img_resized, 0.6, mask_bgr, 0.4, 0)

    # Encode images to base64
    _, buf_sat = cv2.imencode('.jpg', img_resized)
    _, buf_mask = cv2.imencode('.jpg', (mask_display * 255).astype(np.uint8))
    _, buf_overlay = cv2.imencode('.jpg', overlay)
    b64_sat     = 'data:image/jpeg;base64,' + base64.b64encode(buf_sat).decode()
    b64_mask    = 'data:image/jpeg;base64,' + base64.b64encode(buf_mask).decode()
    b64_overlay = 'data:image/jpeg;base64,' + base64.b64encode(buf_overlay).decode()

    total_pixels = mask_original.size
    water_pixels_cnt = int(np.sum(mask_original))
    water_percent = round((water_pixels_cnt / total_pixels) * 100, 2)

    # GIS area calculation — use original-size mask with original transform
    try:
        area_m2, water_pixels_geo, reproj_mask_path = calculate_area(mask_original, transform, crs)
    except Exception as e:
        area_m2 = 0
        water_pixels_geo = water_pixels_cnt
        reproj_mask_path = None

    # Vectorize → GeoJSON
    geojson_content = None
    zip_path = None
    if reproj_mask_path:
        try:
            geojson_path, zip_path = mask_to_vector(reproj_mask_path, uid)
            with open(geojson_path, 'r') as f:
                geojson_content = json.load(f)
        except Exception:
            pass

    result = {
        'uid': uid,
        'place_name': place_name,
        'lat': lat,
        'lon': lon,
        'water_area_m2': round(area_m2, 2),
        'water_area_ha': round(area_m2 / 10000, 4),
        'water_area_km2': round(area_m2 / 1e6, 6),
        'water_pixels': water_pixels_cnt,
        'status_logs': ['🧠 Running U-Net water segmentation…', 'Processing 512×512 satellite patch'],
        'satellite_b64': b64_sat,
        'mask_b64': b64_mask,
        'overlay_b64': b64_overlay,
        'geojson': geojson_content,
    }
    if zip_path:
        result['shapefile_zip_url'] = f'/exports/{os.path.basename(zip_path)}'

    return jsonify(result)

# Serve exported files
@app.route('/exports/<path:filename>')
def serve_export(filename):
    return send_from_directory(EXPORTS_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
