import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_unet

# 1. Enforce GPU usage
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPU detected by TensorFlow! Ensure CUDA, cuDNN, and the correct TensorFlow GPU version are installed. Exiting because you requested strictly GPU training.")

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {gpus}")
except RuntimeError as e:
    print(e)

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR = os.path.join(DATASET_ROOT, 'Masks')

IMAGE_SIZE = (256, 256)
EPOCHS = 3
BATCH_SIZE = 8
SUBSET_SIZE = 1000  # Train on exactly 1000 images

def list_pairs():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.jpg')])

    image_stems = {os.path.splitext(f)[0] for f in image_files}
    mask_stems = {os.path.splitext(f)[0] for f in mask_files}
    common_stems = sorted(image_stems & mask_stems)

    pairs = []
    for stem in common_stems:
        image_path = os.path.join(IMAGE_DIR, stem + '.jpg')
        mask_path = os.path.join(MASK_DIR, stem + '.jpg')
        if os.path.exists(image_path) and os.path.exists(mask_path):
            pairs.append((image_path, mask_path))
    return pairs

def load_pair(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask

if __name__ == '__main__':
    pairs = list_pairs()
    
    if len(pairs) < SUBSET_SIZE:
        print(f"Warning: Only found {len(pairs)} images, which is less than requested 1000.")
    else:
        pairs = pairs[:SUBSET_SIZE]

    print(f"Loading {len(pairs)} images into memory...")

    images, masks = [], []
    for img_path, mask_path in pairs:
        img, msk = load_pair(img_path, mask_path)
        images.append(img)
        masks.append(msk)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # 2. Force the model to be built and trained exclusively on the GPU
    with tf.device('/GPU:0'):
        print("Starting model training strictly on GPU:0...")
        model = build_unet(input_shape=(256, 256, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    model.save(os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5'))
    print('✅ GPU Training completed. Model saved to models/water_unet.h5')
