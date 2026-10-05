import json
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

# A small distance flags images with very similar brightness patterns.
MAX_DISTANCE = 4


def difference_hash(path):
    """Represent horizontal brightness changes as a 64-bit integer."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = [
            image.getpixel((column, row)) for row in range(8) for column in range(9)
        ]

    value = 0

    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)

    return value


def main():
    exclusion_file = PROJECT_ROOT / "training" / "exclusions.json"
    exclusions = set(json.loads(exclusion_file.read_text(encoding="utf-8"))["paths"])

    records = []

    for category in CLASSES:
        folder = DATA_DIR / category

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing category folder: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            relative_path = path.relative_to(PROJECT_ROOT).as_posix()

            if relative_path not in exclusions:
                records.append(
                    {
                        "path": relative_path,
                        "category": category,
                        "hash": difference_hash(path),
                    }
                )

    if not records:
        raise ValueError("No images found. Check data/raw.")

    print(f"Comparing {len(records)} images...")
    candidates = []

    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            distance = (first["hash"] ^ second["hash"]).bit_count()

            if distance <= MAX_DISTANCE:
                candidates.append(
                    {
                        "image_1": first["path"],
                        "image_2": second["path"],
                        "distance": distance,
                        "same_category": (first["category"] == second["category"]),
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["distance"],
            item["image_1"],
            item["image_2"],
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "similarity_candidates.json"
    output.write_text(
        json.dumps(
            {
                "method": "64-bit difference hash",
                "max_distance": MAX_DISTANCE,
                "images_checked": len(records),
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Candidate pairs:", len(candidates))

    for pair in candidates[:10]:
        print(
            f"Distance {pair['distance']}: " f"{pair['image_1']} <-> {pair['image_2']}"
        )

    print("Full report saved to:", output)


if __name__ == "__main__":
    main()
