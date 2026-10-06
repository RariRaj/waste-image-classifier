import csv
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def read_manifest(split_name):
    """Read image paths and numeric labels from an existing split."""
    manifest_path = SPLIT_DIR / f"{split_name}.csv"

    paths = []
    labels = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = PROJECT_ROOT / row["path"]

            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            paths.append(str(image_path))
            labels.append(int(row["label_id"]))

    if not paths:
        raise ValueError(f"No images found in {manifest_path}")

    return paths, labels


def load_image(path, label):
    """Decode one image and resize it to the CNN input size."""
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    # Keep pixels approximately in the 0–255 range.
    # The CNN will contain a Rescaling layer.
    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(split_name, training=False):
    """Build a batched TensorFlow dataset."""
    paths, labels = read_manifest(split_name)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            tf.constant(labels, dtype=tf.int32),
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


def main():
    train_dataset, train_labels = make_dataset("train", training=True)
    val_dataset, val_labels = make_dataset("val")

    print(f"Training images: {len(train_labels)}")
    print(f"Validation images: {len(val_labels)}")

    for name, dataset in [
        ("Training", train_dataset),
        ("Validation", val_dataset),
    ]:
        images, labels = next(iter(dataset))

        print(f"\n{name} batch:")
        print("Image shape:", images.shape)
        print("Label shape:", labels.shape)
        print("Image dtype:", images.dtype)
        print(
            "Pixel range:",
            float(tf.reduce_min(images).numpy()),
            "to",
            float(tf.reduce_max(images).numpy()),
        )
        print("First 10 labels:", labels[:10].numpy())


if __name__ == "__main__":
    main()
