import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from training.cnn_data import (
    IMAGE_SIZE,
    PROJECT_ROOT,
    SEED,
    SPLIT_DIR,
    make_dataset,
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = ARTIFACT_DIR / "mobilenet_v2_frozen.keras"

MAX_EPOCHS = 20


def build_model(number_of_classes):
    """Train a new classifier on frozen ImageNet features."""
    layers = tf.keras.layers

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Preserve the pretrained feature-extraction weights.
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # Active during training only.
    x = layers.RandomFlip("horizontal", seed=SEED)(inputs)
    x = layers.RandomRotation(0.05, seed=SEED + 1)(x)
    x = layers.RandomZoom(0.1, seed=SEED + 2)(x)

    # MobileNetV2 expects pixels approximately between -1 and 1.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
        name="mobilenet_preprocessing",
    )(x)

    # Keep the pretrained batch-normalization layers in inference mode.
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3, seed=SEED + 3)(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="waste_classifier",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenet_v2_frozen",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class ValidationMacroF1(tf.keras.callbacks.Callback):
    """Add validation macro F1 to the epoch logs."""

    def __init__(self, dataset, labels, label_ids):
        super().__init__()
        self.dataset = dataset
        self.labels = np.asarray(labels)
        self.label_ids = label_ids
        self.scores = []

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.dataset, verbose=0)
        predictions = probabilities.argmax(axis=1)

        score = float(
            f1_score(
                self.labels,
                predictions,
                labels=self.label_ids,
                average="macro",
                zero_division=0,
            )
        )

        self.scores.append(score)

        if logs is not None:
            logs["val_macro_f1"] = score

        print(f"\nValidation macro F1: {score:.4f}")


def main():
    tf.keras.utils.set_random_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = json.loads((SPLIT_DIR / "split_summary.json").read_text(encoding="utf-8"))
    class_names = summary["class_names"]
    label_ids = list(range(len(class_names)))

    train_dataset, train_labels = make_dataset("train", training=True)
    val_dataset, val_labels = make_dataset("val")

    # Separate, unshuffled dataset for final training evaluation.
    train_eval_dataset, train_eval_labels = make_dataset("train")

    if sorted(set(train_labels)) != label_ids:
        raise ValueError("Training labels do not match the expected class IDs.")

    if not set(val_labels).issubset(set(label_ids)):
        raise ValueError("Validation contains unexpected class IDs.")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(label_ids),
        y=np.asarray(train_labels),
    )
    class_weights = {
        label_id: float(weight) for label_id, weight in zip(label_ids, weights)
    }

    print("Classes:", class_names)
    print("Class weights:", class_weights)

    model = build_model(len(class_names))
    model.summary()

    macro_f1_callback = ValidationMacroF1(
        val_dataset,
        val_labels,
        label_ids,
    )

    # Order matters: calculate val_macro_f1 before callbacks use it.
    callbacks = [
        macro_f1_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )

    # Evaluate the saved best epoch, not necessarily the final epoch.
    best_model = tf.keras.models.load_model(MODEL_PATH)

    train_predictions = best_model.predict(train_eval_dataset, verbose=0).argmax(axis=1)

    val_predictions = best_model.predict(val_dataset, verbose=0).argmax(axis=1)

    training_accuracy = float(accuracy_score(train_eval_labels, train_predictions))
    validation_accuracy = float(accuracy_score(val_labels, val_predictions))
    validation_macro_f1 = float(
        f1_score(
            val_labels,
            val_predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )

    best_epoch = int(np.argmax(macro_f1_callback.scores) + 1)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Training accuracy:   {training_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Validation macro F1: {validation_macro_f1:.4f}")

    print("\nValidation classification report:")
    print(
        classification_report(
            val_labels,
            val_predictions,
            labels=label_ids,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    history_values = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    history_values["val_macro_f1"] = macro_f1_callback.scores

    report = {
        "model": "mobilenet_v2_frozen",  # Changed
        "pretrained_weights": "imagenet",  # Added
        "base_model_trainable": False,  # Added
        "preprocessing": "pixel / 127.5 - 1",  # Added
        "max_epochs": MAX_EPOCHS,  # Added
        "tensorflow_version": tf.__version__,
        "seed": SEED,
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "input_pixel_range": [0, 255],
        "selection_metric": "validation_macro_f1",
        "best_epoch": best_epoch,
        "epochs_completed": len(macro_f1_callback.scores),
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "class_weights": class_weights,
        "validation_classification_report": classification_report(
            val_labels,
            val_predictions,
            labels=label_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix": confusion_matrix(
            val_labels,
            val_predictions,
            labels=label_ids,
        ).tolist(),
        "history": history_values,
    }

    report_path = REPORT_DIR / "mobilenet_v2_frozen_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/mobilenet_v2_frozen.keras")

    print("Saved: reports/mobilenet_v2_frozen_metrics.json")


if __name__ == "__main__":
    main()
