import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_unet

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is available and will be used: {physical_devices}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Training will fall back to CPU.")

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR = os.path.join(DATASET_ROOT, 'Masks')

IMAGE_SIZE = (256, 256)
EPOCHS = 1  # Just 1 epoch for a quick test
BATCH_SIZE = 4
SUBSET_SIZE = 40  # Only use 40 images total

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
    
    # Grab only a small subset of images
    pairs = pairs[:SUBSET_SIZE]
    print(f"Training on a fast subset of {len(pairs)} images...")

    images, masks = [], []
    for img_path, mask_path in pairs:
        img, msk = load_pair(img_path, mask_path)
        images.append(img)
        masks.append(msk)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    model = build_unet(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    model.save(os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5'))
    print('Fast training completed. Model saved to models/water_unet.h5')
