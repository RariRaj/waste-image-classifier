from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.feature import hog

# A compact starting resolution for the traditional ML baseline.
IMAGE_SIZE = (128, 128)
COLOR_BINS = 32


def extract_features(image_path: str | Path) -> np.ndarray:
    """Convert one image into a fixed-length HOG + color feature vector."""

    # 1. Open, correct orientation, convert to RGB, and resize.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            resample=Image.Resampling.LANCZOS,
        )

        # Convert pixels from integers in [0, 255] to floats in [0, 1].
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    # 2. Extract edge and shape information from grayscale.
    gray = rgb2gray(rgb)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=None,
    )

    # 3. Calculate one normalized histogram per RGB channel.
    color_features = []

    for channel in range(3):
        histogram, _ = np.histogram(
            rgb[:, :, channel],
            bins=COLOR_BINS,
            range=(0.0, 1.0),
        )

        histogram = histogram.astype(np.float32)
        histogram /= histogram.sum()

        color_features.append(histogram)

    # 4. Join all features into one vector.
    features = np.concatenate([hog_features, *color_features]).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features found: {image_path}")

    return features
