"""Upload butterfly dataset to HuggingFace Hub as parquet.

Usage:
    uv run python upload.py --repo-id YOUR_USERNAME/cambridge-butterflies
"""

import pathlib

import beartype
import datasets
import polars as pl
import tqdm
from PIL import Image


@beartype.beartype
def main(
    repo_id: str,
    dataset_dpath: pathlib.Path = pathlib.Path(
        "/local/scratch/stevens.994/datasets/cambridge-segfolder"
    ),
    split: str = "training",
):
    """Upload dataset to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'username/dataset-name').
        dataset_dpath: Path to the dataset directory.
        split: Split name to upload.
    """
    labels_fpath = dataset_dpath / "labels.csv"
    images_dpath = dataset_dpath / "images" / split
    annotations_dpath = dataset_dpath / "annotations" / split

    # Read labels
    df = pl.read_csv(labels_fpath)

    # Build dataset rows
    rows = []
    for row in tqdm.tqdm(df.iter_rows(named=True), total=len(df), desc="Loading"):
        stem = row["stem"]
        img_fpath = images_dpath / f"{stem}.jpg"
        mask_fpath = annotations_dpath / f"{stem}.png"

        if not img_fpath.exists() or not mask_fpath.exists():
            continue

        rows.append({
            "image": Image.open(img_fpath),
            "mask": Image.open(mask_fpath),
            "stem": stem,
            "dataset": row["dataset"],
            "subspecies": row["subspecies"],
            "view": row["view"],
        })

    # Create HuggingFace dataset
    ds = datasets.Dataset.from_list(rows)

    # Push to hub
    print(f"Uploading {len(ds)} samples to {repo_id}...")
    ds.push_to_hub(repo_id, private=True)
    print(f"Done! Dataset available at https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
