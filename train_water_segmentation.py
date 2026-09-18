import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Water Bodies Dataset'))
IMAGE_DIR = os.path.join(DATASET_ROOT, 'Images')
MASK_DIR = os.path.join(DATASET_ROOT, 'Masks')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)


def list_image_pairs():
    image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))])
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if os.path.isfile(os.path.join(MASK_DIR, f))])

    image_stems = {os.path.splitext(os.path.basename(p))[0] for p in image_files}
    mask_stems = {os.path.splitext(os.path.basename(p))[0] for p in mask_files}
    common_stems = sorted(image_stems & mask_stems)

    pairs = []
    for stem in common_stems:
        image_path = os.path.join(IMAGE_DIR, stem + '.jpg')
        mask_path = os.path.join(MASK_DIR, stem + '.jpg')
        if os.path.exists(image_path) and os.path.exists(mask_path):
            pairs.append((image_path, mask_path))
    return pairs


def load_and_preprocess(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f'Could not read image or mask: {image_path}, {mask_path}')

    image = cv2.resize(image, IMAGE_SIZE)
    mask = cv2.resize(mask, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask


def build_dataset():
    pairs = list_image_pairs()
    if not pairs:
        raise FileNotFoundError('No matching image-mask pairs were found in the dataset folders.')

    images = []
    masks = []
    for image_path, mask_path in tqdm(pairs, desc='Loading dataset'):
        image, mask = load_and_preprocess(image_path, mask_path)
        images.append(image)
        masks.append(mask)

    images = np.stack(images, axis=0)
    masks = np.stack(masks, axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    X_train, X_val, X_test, y_train, y_val, y_test = build_dataset()
    print('Training samples:', X_train.shape[0])
    print('Validation samples:', X_val.shape[0])
    print('Test samples:', X_test.shape[0])

    sample_image = X_train[0]
    sample_mask = y_train[0]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title('Sample Image')
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask[:, :, 0], cmap='gray')
    plt.title('Sample Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_data.png'))
    plt.close()

    print('Dataset preparation completed successfully.')
    print('Next: install TensorFlow and replace this script with a full U-Net training loop.')
