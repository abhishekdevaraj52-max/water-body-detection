import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

print("===============================================================")
print("Starting CPU training for the NEXT 1000 images (images 1000 to 2000).")
print("===============================================================")

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR = os.path.join(DATASET_ROOT, 'Masks')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5')

IMAGE_SIZE = (256, 256)
EPOCHS = 1
BATCH_SIZE = 8
START_INDEX = 1000
END_INDEX = 2000

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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find existing model at {MODEL_PATH}. Please run the first training script first!")

    pairs = list_pairs()
    
    # Grab the next 1000 images
    pairs = pairs[START_INDEX:END_INDEX]
    
    if len(pairs) == 0:
        raise ValueError("Not enough images in the dataset to train the next 1000!")

    print(f"Loading {len(pairs)} images into memory...")

    images, masks = [], []
    for img_path, mask_path in pairs:
        img, msk = load_pair(img_path, mask_path)
        images.append(img)
        masks.append(msk)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    print("Loading existing model to continue training...")
    model = load_model(MODEL_PATH)
    
    # Recompile just in case (optional, but good practice)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Starting training loop...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    model.save(MODEL_PATH)
    print('✅ Training completed. Model updated and saved to models/water_unet.h5')
