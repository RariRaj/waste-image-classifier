from collections import defaultdict
from pathlib import Path
import hashlib
import json

from PIL import Image

# Resolve paths relative to this script, regardless of terminal location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_dataset():
    counts = {}
    invalid_images = []
    hashes = defaultdict(list)
    image_sizes = defaultdict(int)

    for class_name in CLASSES:
        folder = DATA_DIR / class_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing category folder: {folder}")

        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )

        counts[class_name] = len(image_paths)

        for path in image_paths:
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()

            try:
                # Check file integrity.
                with Image.open(path) as image:
                    image.verify()

                # Reopen and decode pixels to catch additional errors.
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size

                image_sizes[f"{width}x{height}"] += 1

                # Matching hashes identify byte-for-byte duplicate files.
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[file_hash].append(relative_path)

            except (OSError, ValueError, Image.DecompressionBombError) as error:
                invalid_images.append(
                    {
                        "path": relative_path,
                        "error": str(error),
                    }
                )

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]

    report = {
        "class_counts": counts,
        "total_images": sum(counts.values()),
        "valid_images": sum(image_sizes.values()),
        "image_sizes": dict(image_sizes),
        "invalid_images": invalid_images,
        "exact_duplicate_groups": duplicate_groups,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dataset_inspection.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Class counts:", counts)
    print("Total images:", report["total_images"])
    print("Valid images:", report["valid_images"])
    print("Image sizes:", report["image_sizes"])
    print("Invalid images:", len(invalid_images))
    print("Exact duplicate groups:", len(duplicate_groups))
    print("Report saved to:", report_path)


if __name__ == "__main__":
    inspect_dataset()
