import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from shared.ml_features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_features(split, class_names):
    """Read a manifest and extract features for each listed image."""
    manifest = SPLIT_DIR / f"{split}.csv"

    with manifest.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")

    features = []
    labels = []

    for index, row in enumerate(rows, start=1):
        label_id = int(row["label_id"])

        if (
            not 0 <= label_id < len(class_names)
            or class_names[label_id] != row["label"]
        ):
            raise ValueError(f"Invalid label mapping: {row}")

        image_path = PROJECT_ROOT / row["path"]

        features.append(extract_features(image_path))
        labels.append(label_id)

        if index % 200 == 0 or index == len(rows):
            print(f"{split}: processed {index}/{len(rows)} images")

    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
    )


def main():
    summary = json.loads((SPLIT_DIR / "split_summary.json").read_text(encoding="utf-8"))
    class_names = summary["class_names"]
    label_ids = list(range(len(class_names)))

    print("Extracting training features...")
    X_train, y_train = load_features("train", class_names)

    print("\nExtracting validation features...")
    X_val, y_val = load_features("val", class_names)

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Fit scaling and classification together using training data only.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    # predict() applies the already-fitted scaler.
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_accuracy = accuracy_score(y_train, train_predictions)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )

    print(f"\nTraining accuracy:   {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")

    print("\nValidation classification report:")
    print(
        classification_report(
            y_val,
            val_predictions,
            labels=label_ids,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    metrics = {
        "model": "HOG + RGB histograms + RBF SVM",
        "training_images": len(y_train),
        "validation_images": len(y_val),
        "feature_count": X_train.shape[1],
        "class_names": class_names,
        "parameters": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
        },
        "training_accuracy": float(train_accuracy),
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(val_macro_f1),
        "validation_classification_report": classification_report(
            y_val,
            val_predictions,
            labels=label_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix": confusion_matrix(
            y_val,
            val_predictions,
            labels=label_ids,
        ).tolist(),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": model,
            "class_names": class_names,
            "feature_count": X_train.shape[1],
            "feature_extractor": "shared.ml_features.extract_features",
        },
        ARTIFACT_DIR / "svm_baseline.joblib",
    )

    (REPORT_DIR / "svm_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/svm_baseline.joblib")
    print("Saved: reports/svm_baseline_metrics.json")


if __name__ == "__main__":
    main()
