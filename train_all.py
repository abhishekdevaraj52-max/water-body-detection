import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from model import build_unet

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR    = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR     = os.path.join(DATASET_ROOT, 'Masks')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5')

IMAGE_SIZE   = (256, 256)
EPOCHS       = 2          # epochs per chunk
BATCH_SIZE   = 8
CHUNK_SIZE   = 500        # images loaded into RAM at a time (tune if RAM is tight)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def list_pairs():
    """Return sorted list of (image_path, mask_path) for all matched pairs."""
    img_stems  = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')}
    mask_stems = {os.path.splitext(f)[0] for f in os.listdir(MASK_DIR)  if f.endswith('.jpg')}
    common     = sorted(img_stems & mask_stems)
    pairs = []
    for stem in common:
        ip = os.path.join(IMAGE_DIR, stem + '.jpg')
        mp = os.path.join(MASK_DIR,  stem + '.jpg')
        if os.path.exists(ip) and os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs

def load_pair(image_path, mask_path):
    image = cv2.imread(image_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMAGE_SIZE).astype(np.float32) / 255.0
    mask  = cv2.resize(mask,  IMAGE_SIZE)
    mask  = (mask > 0).astype(np.float32)
    mask  = np.expand_dims(mask, axis=-1)
    return image, mask

def load_chunk(pairs):
    images, masks = [], []
    for ip, mp in pairs:
        img, msk = load_pair(ip, mp)
        images.append(img)
        masks.append(msk)
    return np.stack(images), np.stack(masks)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  FULL DATASET TRAINING - Water Body Segmentation (U-Net)")
    print("=" * 65)

    all_pairs  = list_pairs()
    total      = len(all_pairs)
    already    = 2000          # how many images were trained in previous runs
                               # (1000 from train_cpu_1000 + 1000 from train_cpu_next_1000)
                               # Set to 0 to retrain everything from scratch.
    remaining  = all_pairs[already:]

    print(f"\n  Dataset summary")
    print(f"       Total matched pairs  : {total}")
    print(f"       Already trained      : {already}")
    print(f"       Remaining to train   : {len(remaining)}")
    print(f"       Chunk size           : {CHUNK_SIZE}")
    print(f"       Epochs per chunk     : {EPOCHS}")
    print(f"       Batch size           : {BATCH_SIZE}")
    print()

    if len(remaining) == 0:
        print("All images have already been trained! Nothing left to do.")
        raise SystemExit(0)

    # Load model
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} ...")
        model = load_model(MODEL_PATH)
    else:
        print("No existing model found - building a fresh U-Net ...")
        model = build_unet(input_shape=(256, 256, 3))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model ready.\n")

    # Chunk-based training
    num_chunks = (len(remaining) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end   = min(start + CHUNK_SIZE, len(remaining))
        chunk = remaining[start:end]

        global_start = already + start + 1
        global_end   = already + end

        print("-" * 65)
        print(f"  Chunk {chunk_idx + 1}/{num_chunks}  |  "
              f"Images {global_start}-{global_end} of {total}  "
              f"({len(chunk)} images)")
        print("-" * 65)

        images, masks = load_chunk(chunk)
        X_train, X_val, y_train, y_val = train_test_split(
            images, masks, test_size=0.2, random_state=42)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        # Save after every chunk so progress is never lost
        model.save(MODEL_PATH)
        trained_so_far = global_end
        still_left     = total - trained_so_far
        print(f"\n  Model saved.  "
              f"Trained so far: {trained_so_far}/{total}  |  "
              f"Remaining: {still_left}\n")

    print("=" * 65)
    print(f"  All {total} images trained successfully!")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 65)
