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

SOURCE_MODEL_PATH = ARTIFACT_DIR / "mobilenet_v2_frozen.keras"
MODEL_PATH = ARTIFACT_DIR / "mobilenet_v2_finetuned.keras"

MAX_EPOCHS = 15
FINE_TUNE_LEARNING_RATE = 0.00001


def build_model(number_of_classes):
    """Load the trained classifier and unfreeze later feature layers."""
    if not SOURCE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {SOURCE_MODEL_PATH}")

    model = tf.keras.models.load_model(
        SOURCE_MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != number_of_classes:
        raise ValueError("Model output does not match class count.")

    # This name comes from your model summary.
    base_model = model.get_layer("mobilenetv2_1.00_128")

    # Enable the base model, then selectively freeze its layers.
    base_model.trainable = True

    start_layer = "block_13_expand"
    layer_names = [layer.name for layer in base_model.layers]

    if start_layer not in layer_names:
        raise ValueError(f"Layer not found: {start_layer}")

    start_index = layer_names.index(start_layer)

    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= start_index and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    # The loaded model retains its original preprocessing,
    # augmentation, classifier, and base-model training=False call.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Fine-tuning from layer: {start_layer}")
    print("Batch-normalization layers remain frozen.")

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
    starting_predictions = model.predict(val_dataset, verbose=0).argmax(axis=1)

    starting_macro_f1 = float(
        f1_score(
            val_labels,
            starting_predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )

    print(f"\nStarting checkpoint validation macro F1: " f"{starting_macro_f1:.4f}")

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
        "model": "mobilenet_v2_finetuned",
        "pretrained_weights": "imagenet",
        "source_checkpoint": SOURCE_MODEL_PATH.name,
        "base_model_trainable": True,
        "fine_tune_from_layer": "block_13_expand",
        "batch_normalization_trainable": False,
        "learning_rate": FINE_TUNE_LEARNING_RATE,
        "starting_validation_macro_f1": starting_macro_f1,
        "macro_f1_change_vs_frozen": (validation_macro_f1 - starting_macro_f1),
        "preprocessing": "pixel / 127.5 - 1",
        "max_epochs": MAX_EPOCHS,
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

    report_path = REPORT_DIR / "mobilenet_v2_finetuned_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/mobilenet_v2_finetuned.keras")

    print("Saved: reports/mobilenet_v2_finetuned_metrics.json")
    print(
        "Macro F1 change versus frozen model: "
        f"{validation_macro_f1 - starting_macro_f1:+.4f}"
    )


if __name__ == "__main__":
    main()
