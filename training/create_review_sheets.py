import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "similarity_review"

PAIRS_PER_PAGE = 8
CELL_WIDTH = 360
ROW_HEIGHT = 230


def main():
    report_path = REPORT_DIR / "similarity_candidates.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = report["candidates"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(candidates), PAIRS_PER_PAGE):
        page_pairs = candidates[start : start + PAIRS_PER_PAGE]

        sheet = Image.new(
            "RGB",
            (CELL_WIDTH * 2, ROW_HEIGHT * len(page_pairs)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for row, pair in enumerate(page_pairs):
            pair_number = start + row + 1
            top = row * ROW_HEIGHT

            for column, key in enumerate(["image_1", "image_2"]):
                path = PROJECT_ROOT / pair[key]
                left = column * CELL_WIDTH

                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    thumbnail = ImageOps.contain(image, (340, 180))
                    sheet.paste(
                        thumbnail,
                        (
                            left + (CELL_WIDTH - thumbnail.width) // 2,
                            top + 40,
                        ),
                    )

                label = f"{path.parent.name}/{path.name}"
                draw.text(
                    (left + 10, top + 5),
                    f"Pair {pair_number} | Distance {pair['distance']}",
                    fill="black",
                )
                draw.text(
                    (left + 10, top + 21),
                    label,
                    fill="black",
                )

            draw.line(
                (0, top + ROW_HEIGHT - 1, CELL_WIDTH * 2, top + ROW_HEIGHT - 1),
                fill="gray",
            )

        page_number = start // PAIRS_PER_PAGE + 1
        output_path = OUTPUT_DIR / f"review_{page_number:02d}.jpg"
        sheet.save(output_path, quality=90)

    page_count = (len(candidates) + PAIRS_PER_PAGE - 1) // PAIRS_PER_PAGE

    print(f"Created {page_count} review sheets.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
