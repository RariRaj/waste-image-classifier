import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SPLITS = ["train", "val", "test"]
RATIOS = [0.70, 0.15, 0.15]
SEED = 42


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    exclusions = set(
        read_json(PROJECT_ROOT / "training" / "exclusions.json")["paths"]
    )
    related_groups = read_json(
        PROJECT_ROOT / "training" / "related_groups.json"
    )["groups"]

    # Inventory the original dataset.
    all_paths = {}

    for label in CLASSES:
        folder = DATA_DIR / label

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }:
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                all_paths[relative] = label

    missing_exclusions = exclusions - set(all_paths)

    if missing_exclusions:
        raise ValueError(
            f"Excluded files not found: {sorted(missing_exclusions)}"
        )

    records = {
        path: label
        for path, label in all_paths.items()
        if path not in exclusions
    }

    if len(records) != 2521:
        raise ValueError(
            f"Expected 2521 usable images, found {len(records)}. "
            "Check the dataset and exclusions."
        )

    # Each image starts in its own group.
    # Union-find merges related groups, including overlapping ones.
    parent = {path: path for path in records}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first, second):
        first_root = find(first)
        second_root = find(second)

        if first_root != second_root:
            parent[second_root] = first_root

    for group in related_groups:
        if len(group) < 2:
            raise ValueError(f"Group must contain at least two files: {group}")

        for path in group:
            if path not in records:
                raise ValueError(f"Grouped file missing or excluded: {path}")

        for path in group[1:]:
            union(group[0], path)

    # Recheck exact hashes and group any remaining identical files.
    hashes = {}
    first_for_hash = {}

    for path in sorted(records):
        digest = hashlib.sha256(
            (PROJECT_ROOT / path).read_bytes()
        ).hexdigest()
        hashes[path] = digest

        if digest in first_for_hash:
            union(path, first_for_hash[digest])
        else:
            first_for_hash[digest] = path

    grouped = defaultdict(list)

    for path in sorted(records):
        grouped[find(path)].append(path)

    groups_by_class = defaultdict(list)

    for members in grouped.values():
        labels = {records[path] for path in members}

        if len(labels) != 1:
            raise ValueError(
                "A related group has conflicting labels. "
                f"Review before splitting: {members}"
            )

        label = next(iter(labels))
        groups_by_class[label].append(sorted(members))

    # Allocate whole groups within each class.
    # Large groups go first; seeded shuffling randomizes equal-size groups.
    rng = random.Random(SEED)
    rows = {split: [] for split in SPLITS}

    for label in CLASSES:
        groups = groups_by_class[label]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

        total = sum(len(group) for group in groups)
        targets = [total * ratio for ratio in RATIOS]
        counts = [0, 0, 0]

        for members in groups:
            # Choose the assignment with the smallest increase
            # in squared deviation from the target counts.
            size = len(members)
            chosen = min(
                range(3),
                key=lambda index: (
                    (counts[index] + size - targets[index]) ** 2
                    - (counts[index] - targets[index]) ** 2
                ),
            )

            split = SPLITS[chosen]
            group_id = members[0]

            for path in members:
                rows[split].append({
                    "path": path,
                    "label": label,
                    "label_id": CLASSES.index(label),
                    "group_id": group_id,
                    "sha256": hashes[path],
                })

            counts[chosen] += size

    # Verify completeness and separation before saving.
    assigned_paths = [
        row["path"]
        for split in SPLITS
        for row in rows[split]
    ]

    if (
        len(assigned_paths) != len(records)
        or set(assigned_paths) != set(records)
    ):
        raise RuntimeError("Some images are missing or assigned twice.")

    for field in ["path", "group_id", "sha256"]:
        seen = set()

        for split in SPLITS:
            values = {row[field] for row in rows[split]}

            if seen & values:
                raise RuntimeError(f"Cross-split overlap detected: {field}")

            seen.update(values)

    for split in SPLITS:
        if {row["label"] for row in rows[split]} != set(CLASSES):
            raise RuntimeError(f"A class is missing from {split}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_files = [
        OUTPUT_DIR / f"{split}.csv" for split in SPLITS
    ] + [OUTPUT_DIR / "split_summary.json"]

    if any(path.exists() for path in output_files):
        raise FileExistsError(
            "Split files already exist. Keep your established split; "
            "do not overwrite it accidentally."
        )

    fields = ["path", "label", "label_id", "group_id", "sha256"]
    summary = {
        "seed": SEED,
        "class_names": CLASSES,
        "excluded_images": len(exclusions),
        "usable_images": len(records),
        "multi_image_groups": sum(
            len(group) > 1 for group in grouped.values()
        ),
        "splits": {},
    }

    print(f"{'Class':<12} {'Train':>7} {'Val':>7} {'Test':>7}")

    class_counts = {
        split: Counter(row["label"] for row in rows[split])
        for split in SPLITS
    }

    for label in CLASSES:
        print(
            f"{label:<12}"
            f"{class_counts['train'][label]:>7}"
            f"{class_counts['val'][label]:>7}"
            f"{class_counts['test'][label]:>7}"
        )

    for split in SPLITS:
        with (OUTPUT_DIR / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                sorted(rows[split], key=lambda row: row["path"])
            )

        summary["splits"][split] = {
            "total": len(rows[split]),
            "class_counts": dict(class_counts[split]),
        }

    (OUTPUT_DIR / "split_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTotal assigned:", len(assigned_paths))
    print("Related groups and exact hashes do not cross splits.")
    print("Saved manifests to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()