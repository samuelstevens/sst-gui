"""Convert butterfly dataset to ImgSegFolder format for saev.

Creates the directory structure expected by ImgSegFolderDataset:
    root/
        images/
            training/
                {stem}.jpg
        annotations/
            training/
                {stem}.png
        labels.csv

Usage:
    uv run python convert_to_segfolder.py
"""

import csv
import pathlib
import shutil

import beartype
import polars as pl


@beartype.beartype
def main(
    master_csv: pathlib.Path = pathlib.Path(
        "/local/scratch/stevens.994/datasets/cambridge-segmented/master.csv"
    ),
    pred_masks_dpath: pathlib.Path = pathlib.Path(
        "/local/scratch/stevens.994/datasets/cambridge-segmented/pred_masks"
    ),
    output_dpath: pathlib.Path = pathlib.Path(
        "/local/scratch/stevens.994/datasets/cambridge-segfolder"
    ),
    split: str = "training",
):
    """Convert butterfly images and masks to ImgSegFolder format.

    Args:
        master_csv: Path to master.csv with image metadata.
        pred_masks_dpath: Directory containing predicted masks ({pk}.png).
        output_dpath: Output directory for the converted dataset.
        split: Split name (training or validation).
    """
    # Read master CSV
    df = pl.read_csv(master_csv, infer_schema_length=None)

    # Create output directories
    images_dpath = output_dpath / "images" / split
    annotations_dpath = output_dpath / "annotations" / split
    images_dpath.mkdir(parents=True, exist_ok=True)
    annotations_dpath.mkdir(parents=True, exist_ok=True)

    # Track which images we process
    processed = []
    skipped_no_mask = 0
    skipped_no_image = 0

    n_total = len(df)
    log_interval = max(1, n_total // 100)  # Log every ~1%

    for i, row in enumerate(df.iter_rows(named=True)):
        if i % log_interval == 0:
            print(f"Progress: {i}/{n_total} ({100*i/n_total:.1f}%)")
        pk = row["Image_name"]
        filepath = row["filepath"]

        # Check if we have a predicted mask for this image
        mask_fpath = pred_masks_dpath / f"{pk}.png"
        if not mask_fpath.exists():
            skipped_no_mask += 1
            continue

        # Build the source image path
        # filepath is like "images/Heliconius melpomene ssp. malleti/19160_19N0010_v.JPG"
        img_fpath = pathlib.Path(
            "/local/scratch/datasets/jiggins/butterflies"
        ) / filepath.replace("images/", "", 1)
        if not img_fpath.exists():
            skipped_no_image += 1
            continue

        # Use pk as the stem (without extension issues)
        stem = pk

        # Determine output paths - keep original extension for images
        img_ext = img_fpath.suffix
        out_img_fpath = images_dpath / f"{stem}{img_ext}"
        out_mask_fpath = annotations_dpath / f"{stem}.png"

        # Copy files
        if not out_img_fpath.exists():
            shutil.copy2(img_fpath, out_img_fpath)
        if not out_mask_fpath.exists():
            shutil.copy2(mask_fpath, out_mask_fpath)

        # Collect labels for CSV
        processed.append({
            "stem": stem,
            "dataset": row["Dataset"] or "unknown",
            "subspecies": row["subspecies"] or "unknown",
            "view": row["View"] or "unknown",
        })

    # Write labels.csv
    labels_fpath = output_dpath / "labels.csv"
    with open(labels_fpath, "w", newline="") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=["stem", "dataset", "subspecies", "view"]
        )
        writer.writeheader()
        writer.writerows(processed)

    print(f"Processed {len(processed)} images")
    print(f"Skipped {skipped_no_mask} (no mask), {skipped_no_image} (no image)")
    print(f"Output directory: {output_dpath}")
    print(f"Labels CSV: {labels_fpath}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
