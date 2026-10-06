import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from training.cnn_data import (
    PROJECT_ROOT,
    SPLIT_DIR,
    make_dataset,
)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "mobilenet_v2_finetuned.keras"
REPORT_PATH = PROJECT_ROOT / "reports" / "mobilenet_v2_test_metrics.json"
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "mobilenet_v2_finetuned_metrics.json"
)


def main():
    # Read the class order recorded during model development.
    validation_report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    class_names = validation_report["class_names"]
    label_ids = list(range(len(class_names)))

    split_summary = json.loads(
        (SPLIT_DIR / "split_summary.json").read_text(encoding="utf-8")
    )

    if class_names != split_summary["class_names"]:
        raise ValueError("Class order differs between model and dataset.")

    # Keep test images in manifest order.
    test_dataset, test_labels = make_dataset("test")
    test_labels = np.asarray(test_labels)

    if not set(test_labels.tolist()).issubset(set(label_ids)):
        raise ValueError("Unexpected class IDs in the test manifest.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output does not match the class count.")

    # Prediction only: no fit(), weight updates, or augmentation.
    probabilities = model.predict(test_dataset, verbose=1)
    predictions = probabilities.argmax(axis=1)

    if len(predictions) != len(test_labels):
        raise ValueError("Prediction and label counts do not match.")

    accuracy = float(accuracy_score(test_labels, predictions))
    macro_f1 = float(
        f1_score(
            test_labels,
            predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )

    correct = int(np.sum(predictions == test_labels))

    print(f"\nTest images: {len(test_labels)}")
    print(f"Correct predictions: {correct}/{len(test_labels)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")

    print("\nTest classification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=label_ids,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    report = {
        "model": "mobilenet_v2_finetuned",
        "checkpoint": MODEL_PATH.name,
        "tensorflow_version": tf.__version__,
        "class_names": class_names,
        "selection_metric": "validation_macro_f1",
        "selected_validation_macro_f1": validation_report["validation_macro_f1"],
        "test_count": len(test_labels),
        "correct_predictions": correct,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_classification_report": classification_report(
            test_labels,
            predictions,
            labels=label_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "test_confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=label_ids,
        ).tolist(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
