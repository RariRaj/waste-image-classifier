import json
from itertools import product

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from training.train_svm import (
    ARTIFACT_DIR,
    REPORT_DIR,
    SPLIT_DIR,
    load_features,
)


def main():
    summary = json.loads((SPLIT_DIR / "split_summary.json").read_text(encoding="utf-8"))
    class_names = summary["class_names"]
    label_ids = list(range(len(class_names)))

    # Extract once, then reuse for every experiment.
    X_train, y_train = load_features("train", class_names)
    X_val, y_val = load_features("val", class_names)

    combinations = list(
        product(
            [0.1, 1.0, 10.0],
            ["scale", 0.0001, 0.00001],
        )
    )

    results = []
    best_model = None
    best_result = None
    best_predictions = None

    for number, (c_value, gamma_value) in enumerate(combinations, start=1):
        print(
            f"\nExperiment {number}/{len(combinations)}: "
            f"C={c_value}, gamma={gamma_value}",
            flush=True,
        )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=c_value,
                        gamma=gamma_value,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        # Both scaler and SVM learn only from training data.
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        result = {
            "C": c_value,
            "gamma": gamma_value,
            "training_accuracy": float(accuracy_score(y_train, train_predictions)),
            "validation_accuracy": float(accuracy_score(y_val, val_predictions)),
            "validation_macro_f1": float(
                f1_score(
                    y_val,
                    val_predictions,
                    labels=label_ids,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        results.append(result)

        print(
            f"Train accuracy: {result['training_accuracy']:.4f} | "
            f"Val accuracy: {result['validation_accuracy']:.4f} | "
            f"Val macro F1: {result['validation_macro_f1']:.4f}"
        )

        # Keep the first model in the event of an exact score tie.
        if (
            best_result is None
            or result["validation_macro_f1"] > best_result["validation_macro_f1"]
        ):
            best_model = model
            best_result = result
            best_predictions = val_predictions

    print("\nResults ranked by validation macro F1:")

    ranked = sorted(
        results,
        key=lambda item: item["validation_macro_f1"],
        reverse=True,
    )

    for result in ranked:
        print(
            f"C={result['C']:<5} "
            f"gamma={str(result['gamma']):<10} "
            f"macro F1={result['validation_macro_f1']:.4f}"
        )

    print("\nSelected settings:", best_result)
    print("\nSelected model — validation report:")
    print(
        classification_report(
            y_val,
            best_predictions,
            labels=label_ids,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # Compare against the baseline settings within this same run.
    baseline = next(
        result
        for result in results
        if result["C"] == 1.0 and result["gamma"] == "scale"
    )
    improvement = best_result["validation_macro_f1"] - baseline["validation_macro_f1"]
    print(f"Macro F1 change versus baseline: {improvement:+.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": best_model,
            "class_names": class_names,
            "feature_count": X_train.shape[1],
            "feature_extractor": "shared.ml_features.extract_features",
            "selected_parameters": {
                "C": best_result["C"],
                "gamma": best_result["gamma"],
            },
        },
        ARTIFACT_DIR / "svm_tuned.joblib",
    )

    report = {
        "selection_metric": "validation_macro_f1",
        "class_names": class_names,
        "experiments": results,
        "best_result": best_result,
        "macro_f1_change_vs_baseline": improvement,
        "validation_classification_report": classification_report(
            y_val,
            best_predictions,
            labels=label_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "validation_confusion_matrix": confusion_matrix(
            y_val,
            best_predictions,
            labels=label_ids,
        ).tolist(),
    }

    (REPORT_DIR / "svm_tuning_results.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/svm_tuned.joblib")
    print("Saved: reports/svm_tuning_results.json")


if __name__ == "__main__":
    main()
