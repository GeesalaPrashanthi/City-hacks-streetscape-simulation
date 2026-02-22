import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download sidewalk images from GeoJSON metadata.")
    parser.add_argument("--geojson", default="aboveGroundAssets.geojson", help="Input GeoJSON file path.")
    parser.add_argument("--out-dir", default="dataset", help="Output dataset directory.")
    parser.add_argument(
        "--conditions",
        default="Good,Fair,Poor",
        help="Comma-separated condition labels to include.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "balanced"],
        default="full",
        help="full = use all available images, balanced = same count per class.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap per class after mode selection (0 = no cap).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files that already exist in output directory (default: true).",
    )
    return parser.parse_args()


def load_features(geojson_path: Path):
    with geojson_path.open() as f:
        data = json.load(f)
    return data.get("features", [])


def select_features(features, valid_conditions):
    selected = []
    for feature in features:
        props = feature.get("properties", {})
        if props.get("asset_type") != "SIDEWALK":
            continue
        condition = props.get("condition")
        image_url = props.get("image_url")
        feature_id = props.get("feature_id")
        if condition not in valid_conditions:
            continue
        if not image_url:
            continue
        if not feature_id:
            continue
        selected.append(
            {
                "condition": str(condition),
                "image_url": str(image_url),
                "feature_id": str(feature_id),
            }
        )
    return selected


def apply_mode(features, mode, max_per_class):
    by_condition = defaultdict(list)
    for item in features:
        by_condition[item["condition"]].append(item)

    print("Available per class:")
    for condition in sorted(by_condition):
        print(f"  {condition}: {len(by_condition[condition])}")

    if mode == "balanced":
        min_count = min(len(items) for items in by_condition.values())
        print(f"Mode=balanced -> using {min_count} per class")
        limited = []
        for condition, items in by_condition.items():
            limited.extend(items[:min_count])
        by_condition = defaultdict(list)
        for item in limited:
            by_condition[item["condition"]].append(item)
    else:
        print("Mode=full -> using all available images")

    if max_per_class > 0:
        print(f"Applying max-per-class cap: {max_per_class}")
        capped = []
        for condition, items in by_condition.items():
            capped.extend(items[:max_per_class])
        return capped

    final = []
    for items in by_condition.values():
        final.extend(items)
    return final


def download_images(items, out_dir: Path, skip_existing: bool):
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        condition = item["condition"]
        feature_id = item["feature_id"]
        image_url = item["image_url"]

        class_dir = out_dir / condition
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{feature_id}.jpg"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            urllib.request.urlretrieve(image_url, out_path)
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"  ✗ {condition}/{feature_id}.jpg -> {exc}")

        if idx % 100 == 0:
            print(f"  progress: {idx}/{len(items)} (downloaded={downloaded}, skipped={skipped}, failed={failed})")

    return downloaded, skipped, failed


def main():
    args = parse_args()
    geojson_path = Path(args.geojson)
    out_dir = Path(args.out_dir)
    valid_conditions = {c.strip() for c in args.conditions.split(",") if c.strip()}

    print(f"GeoJSON: {geojson_path}")
    print(f"Output: {out_dir}")
    print(f"Conditions: {sorted(valid_conditions)}")

    features = load_features(geojson_path)
    selected = select_features(features, valid_conditions)
    print(f"Total selected before balancing/capping: {len(selected)}")

    final_items = apply_mode(selected, args.mode, args.max_per_class)
    print(f"Final download target: {len(final_items)} images")

    downloaded, skipped, failed = download_images(final_items, out_dir, args.skip_existing)
    print("\nDownload complete")
    print(f"  downloaded: {downloaded}")
    print(f"  skipped:    {skipped}")
    print(f"  failed:     {failed}")


if __name__ == "__main__":
    main()
