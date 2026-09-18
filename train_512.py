import os
import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# GPU CHECK - CPU TRAINING IS NOT ALLOWED
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

if not gpus:
    print("GPU DETECTED : NO")
    print("TRAINING STOPPED")
    print("CPU TRAINING IS NOT ALLOWED")
    print("=" * 60)
    raise SystemExit(1)

print("GPU DETECTED : YES")
print("GPU:", gpus[0])

# Enable GPU memory growth
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("GPU memory configuration error:", e)
        raise

print("GPU training enabled.")
print("=" * 60)
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_unet

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR    = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR     = os.path.join(DATASET_ROOT, 'Masks')
MODELS_DIR   = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH   = os.path.join(MODELS_DIR, 'water_unet.h5')
BACKUP_PATH  = os.path.join(MODELS_DIR, 'water_unet_backup.h5')

IMAGE_SIZE   = (512, 512)
EPOCHS       = 3
BATCH_SIZE   = 1
SUBSET_SIZE  = None  # Use full dataset

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def list_pairs():
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

# ---------------------------------------------------------------------
#  MEMORY‑EFFICIENT DATA LOADER
# ---------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    """Loads images and masks on‑the‑fly for a given list of file pairs.

    Args:
        pairs (list): List of (image_path, mask_path) tuples.
        batch_size (int): Number of samples per batch.
        image_size (tuple): Desired image dimensions (height, width).
    """
    def __init__(self, pairs, batch_size=1, image_size=(512, 512)):
        self.pairs = pairs
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.pairs))

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        # Generate one batch of data
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_pairs = [self.pairs[i] for i in batch_indexes]
        images, masks = [], []
        for img_path, mask_path in batch_pairs:
            img, msk = load_pair(img_path, mask_path)
            images.append(img)
            masks.append(msk)
        return np.stack(images), np.stack(masks)

    def on_epoch_end(self):
        # Shuffle indexes after each epoch for better training
        np.random.shuffle(self.indexes)


def plot_history(history):
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_accuracy.png')

    # IoU
    if 'iou_metric' in history.history:
        plt.figure()
        plt.plot(history.history['iou_metric'], label='Train IoU')
        plt.plot(history.history['val_iou_metric'], label='Val IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.savefig('training_iou.png')

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  TRAINING AT 512x512 - Water Body Segmentation (U-Net)")
    print("=" * 65)

    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Backup existing model
    if os.path.exists(MODEL_PATH):
        print(f"Creating backup of existing model at {BACKUP_PATH}...")
    try:
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print("Backup created successfully.")
    except PermissionError:
        print("WARNING: Backup could not be created.")
        print("Continuing training without backup.")
    
    all_pairs = list_pairs()
    print(f"Loaded {len(all_pairs)} image/mask pairs for training.")
    # Split into training and validation pairs
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42)
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(val_pairs)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    # Create generators that load data lazily
    train_generator = DataGenerator(train_pairs, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    val_generator = DataGenerator(val_pairs, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    print("Building U-Net model with input shape 512x512x3...")
    model = build_unet(input_shape=(512, 512, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', iou_metric, dice_metric])
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    print(f"Training complete. Generating plots...")
    plot_history(history)
    
    # Final evaluation on validation set
    val_metrics = model.evaluate(val_generator, verbose=1)

    print("\nMODEL TRAINING COMPLETE")
    print("Input Size: 512x512x3")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Exists: {'YES' if os.path.exists(MODEL_PATH) else 'NO'}")
    
    # Verify we can load it
    try:
        from tensorflow.keras.models import load_model
        # Load with custom objects
        loaded_model = load_model(MODEL_PATH, custom_objects={'iou_metric': iou_metric, 'dice_metric': dice_metric})
        print("Model Loaded: YES")
        print(f"Model Parameters: {loaded_model.count_params()}")
    except Exception as e:
        print(f"Model Loaded: NO (Error: {e})")
        print("Model Parameters: 0")

    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {val_metrics[1]:.4f}")
    print(f"IoU: {val_metrics[2]:.4f}")
    print(f"Dice: {val_metrics[3]:.4f}")
