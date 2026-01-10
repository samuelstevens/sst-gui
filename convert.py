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
    uv run python convert.py
"""

import concurrent.futures
import csv
import pathlib

import beartype
import polars as pl
import tqdm
from PIL import Image, ImageOps


@beartype.beartype
def _resize_shortest_side(img: Image.Image, size: int) -> Image.Image:
    """Resize image so the shortest side equals `size`, maintaining aspect ratio."""
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


@beartype.beartype
def _resize_shortest_side_nearest(img: Image.Image, size: int) -> Image.Image:
    """Resize image so the shortest side equals `size`, using NEAREST for masks."""
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    return img.resize((new_w, new_h), Image.NEAREST)


@beartype.beartype
def _process_one(
    img_fpath: pathlib.Path,
    mask_fpath: pathlib.Path,
    out_img_fpath: pathlib.Path,
    out_mask_fpath: pathlib.Path,
    shortest_side: int,
) -> bool:
    """Process a single image and mask pair. Returns True if processed, False if skipped."""
    if out_img_fpath.exists() and out_mask_fpath.exists():
        return False

    # Load and resize image
    if not out_img_fpath.exists():
        img = Image.open(img_fpath)
        img = ImageOps.exif_transpose(img)
        img = _resize_shortest_side(img, shortest_side)
        img.save(out_img_fpath, quality=95)

    # Load and resize mask (use NEAREST to preserve label values)
    if not out_mask_fpath.exists():
        mask = Image.open(mask_fpath)
        mask = _resize_shortest_side_nearest(mask, shortest_side)
        mask.save(out_mask_fpath)

    return True


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
    shortest_side: int = 1024,
    n_workers: int = 16,
):
    """Convert butterfly images and masks to ImgSegFolder format.

    Args:
        master_csv: Path to master.csv with image metadata.
        pred_masks_dpath: Directory containing predicted masks ({pk}.png).
        output_dpath: Output directory for the converted dataset.
        split: Split name (training or validation).
        shortest_side: Resize images so shortest side equals this value.
        n_workers: Number of parallel workers for image processing.
    """
    # Read master CSV
    df = pl.read_csv(master_csv, infer_schema_length=None)

    # Create output directories
    images_dpath = output_dpath / "images" / split
    annotations_dpath = output_dpath / "annotations" / split
    images_dpath.mkdir(parents=True, exist_ok=True)
    annotations_dpath.mkdir(parents=True, exist_ok=True)

    # Build list of jobs
    jobs = []
    processed_metadata = []
    skipped_no_mask = 0
    skipped_no_image = 0

    for row in df.iter_rows(named=True):
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

        # Output paths - always save as .jpg for images
        out_img_fpath = images_dpath / f"{stem}.jpg"
        out_mask_fpath = annotations_dpath / f"{stem}.png"

        jobs.append((
            img_fpath,
            mask_fpath,
            out_img_fpath,
            out_mask_fpath,
            shortest_side,
        ))
        processed_metadata.append({
            "stem": stem,
            "dataset": row["Dataset"] or "unknown",
            "subspecies": row["subspecies"] or "unknown",
            "view": row["View"] or "unknown",
        })

    print(f"Found {len(jobs)} images to process")
    print(f"Skipped {skipped_no_mask} (no mask), {skipped_no_image} (no image)")

    # Process images in parallel
    n_processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_process_one, *job) for job in jobs]
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(futs),
            total=len(futs),
            desc="Processing images",
        ):
            if fut.result():
                n_processed += 1

    # Write labels.csv
    labels_fpath = output_dpath / "labels.csv"
    with open(labels_fpath, "w", newline="") as fd:
        writer = csv.DictWriter(
            fd, fieldnames=["stem", "dataset", "subspecies", "view"]
        )
        writer.writeheader()
        writer.writerows(processed_metadata)

    print(f"Processed {n_processed} new images ({len(jobs)} total in dataset)")
    print(f"Output directory: {output_dpath}")
    print(f"Labels CSV: {labels_fpath}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
