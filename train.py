import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model import build_unet

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR = os.path.join(DATASET_ROOT, 'Masks')

IMAGE_SIZE = (256, 256)
EPOCHS = 5
BATCH_SIZE = 4


def list_pairs():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if os.path.isfile(os.path.join(MASK_DIR, f))])

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
    if image is None or mask is None:
        raise ValueError(f'Could not read: {image_path} or {mask_path}')

    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask


if __name__ == '__main__':
    pairs = list_pairs()
    if not pairs:
        raise FileNotFoundError('No matching images and masks were found.')

    images = []
    masks = []
    for image_path, mask_path in pairs:
        image, mask = load_pair(image_path, mask_path)
        images.append(image)
        masks.append(mask)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    model = build_unet(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    model.save(os.path.join(os.path.dirname(__file__), 'models', 'water_unet.h5'))
    print('Training completed. Model saved to models/water_unet.h5')
