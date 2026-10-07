"""Load the selected model and predict waste categories."""

import json
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "mobilenet_v2_finetuned.keras"
METADATA_PATH = PROJECT_ROOT / "reports" / "mobilenet_v2_finetuned_metrics.json"

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB
MAX_IMAGE_PIXELS = 12_000_000


class InvalidImageError(ValueError):
    """The uploaded file is not a supported, valid image."""


class WastePredictor:
    def __init__(self):
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        self.class_names = metadata["class_names"]
        self.image_size = tuple(metadata["image_size"])
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )
        self.lock = Lock()

        if self.model.output_shape[-1] != len(self.class_names):
            raise ValueError("Model output and class names do not match.")

        expected_shape = (*self.image_size, 3)
        if tuple(self.model.input_shape[1:]) != expected_shape:
            raise ValueError("Model input and image size do not match.")

        # Warm up the model once at startup.
        sample = tf.zeros((1, *self.image_size, 3), dtype=tf.float32)
        self.model(sample, training=False)

    def predict(self, image_bytes):
        # Check actual file contents, not just its filename.
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise InvalidImageError("Only JPEG and PNG images are supported.")

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImageError(
                        "Image dimensions are too large. " "Maximum: 12 million pixels."
                    )

                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImageError("Please upload a non-animated image.")

                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as exc:
            raise InvalidImageError("The file is not a valid readable image.") from exc

        try:
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False,
            )
            image.set_shape([None, None, 3])

            image = tf.image.resize(
                image,
                self.image_size,
                antialias=True,
            )
            image = tf.cast(image, tf.float32)
            batch = tf.expand_dims(image, axis=0)

        except (tf.errors.OpError, ValueError) as exc:
            raise InvalidImageError("The image could not be decoded.") from exc

        # Keep pixels in 0–255: preprocessing is inside the saved model.
        # Serialize inference calls for this initial CPU service.
        with self.lock:
            scores = self.model(batch, training=False).numpy()[0]

        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Model returned invalid scores.")

        best_index = int(np.argmax(scores))

        return {
            "predicted_class": self.class_names[best_index],
            "confidence": float(scores[best_index]),
            "class_scores": {
                name: float(score) for name, score in zip(self.class_names, scores)
            },
            "model": "mobilenet_v2_finetuned",
        }
