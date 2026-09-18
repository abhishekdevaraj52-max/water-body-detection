import os
import time
import cv2
import numpy as np

# Inference size must match the training resolution of the U-Net model (512×512).
IMG_SIZE = 512
WATER_THRESHOLD = 0.5

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset', 'Images', 'water_body_1.jpg')

# ---------------------------------------------------------------------------
# Module-level model cache: load once, reuse on every subsequent call.
# ---------------------------------------------------------------------------
_model = None

def _get_model():
    """Return the cached U-Net model, loading it once if necessary."""
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        t0 = time.perf_counter()
        _model = load_model(MODEL_PATH)
        print(f"[TIMING] Model load: {time.perf_counter() - t0:.2f}s")
        # Warm-up: compile the TF graph on a dummy tensor
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _model(dummy, training=False)
        print(f"[TIMING] Graph warm-up done")
    return _model


def predict_one(image_path):
    t_start = time.perf_counter()

    model = _get_model()
    print("========================================")
    print("U-NET INFERENCE")
    print("========================================")
    print(f"Model input shape: {model.input_shape}")

    # ── Preprocess ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not found: {image_path}')

    original_shape = image.shape
    print(f"Original image shape: {original_shape}")

    # Resize a COPY to 512×512 for U-Net inference
    inference_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    inference_image = inference_image.astype(np.float32) / 255.0
    img_input = np.expand_dims(inference_image, axis=0)
    print(f"[TIMING] Preprocess: {(time.perf_counter()-t0)*1000:.1f} ms  shape={img_input.shape}")

    # ── Inference — direct model call (lower overhead than .predict) ──────
    t0 = time.perf_counter()
    prediction_tensor = model(img_input, training=False)
    prediction = (
        prediction_tensor.numpy()
        if hasattr(prediction_tensor, 'numpy')
        else np.array(prediction_tensor)
    )
    print(f"[TIMING] Inference:  {(time.perf_counter()-t0)*1000:.1f} ms  shape={prediction.shape}")

    # ── Postprocess ────────────────────────────────────────────────
    t0 = time.perf_counter()
    mask_512 = (prediction[0, :, :, 0] >= WATER_THRESHOLD).astype(np.uint8)

    # Resize mask back to original image dimensions
    mask_original = cv2.resize(
        mask_512,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    print(f"[TIMING] Postprocess:{(time.perf_counter()-t0)*1000:.1f} ms  final shape={mask_original.shape}")
    print(f"[TIMING] Total:      {(time.perf_counter()-t_start)*1000:.1f} ms")
    print(f"Water threshold: {WATER_THRESHOLD}")
    print("========================================")

    return mask_original


if __name__ == '__main__':
    mask = predict_one(IMAGE_PATH)
    print('Prediction shape:', mask.shape)
    print('Water pixels:', int(mask.sum()))
