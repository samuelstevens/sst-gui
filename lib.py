import collections.abc
import logging
import pathlib
import time
from typing import Literal

import beartype
import numpy as np
import pydantic


class Spec(pydantic.BaseModel):
    """
    Everything required to do inference. The GUI generates a JSON representation of this object.
    Masks are stored as single-channel (L-mode) .png files with $PRIMARY_KEY.png as the name, where each pixel has the object ID as the value (at most 255 objects).
    """

    root: pathlib.Path
    """Directory with master.csv, and ref_masks/ and pred_masks/ directories."""
    filter_query: str
    """SQL query to filter master_csv."""
    group_by: tuple[str, ...]
    img_path: str
    primary_key: str
    sam2: str
    """SAM2 model name/path."""
    device: str
    """Device to run inference on (cuda/cpu)."""
    mask_mode: Literal["original", "position", "binary"]
    """Mask transformation mode: original, position, or binary."""

    @property
    def master_csv(self) -> pathlib.Path:
        return self.root / "master.csv"

    @property
    def ref_masks(self) -> pathlib.Path:
        return self.root / "ref_masks"

    @property
    def pred_masks(self) -> pathlib.Path:
        return self.root / "pred_masks"


@beartype.beartype
def get_obj_ids(mask: np.ndarray) -> list[int]:
    """Return sorted object IDs in a mask (excluding background)."""
    return [int(x) for x in np.unique(mask) if x > 0]


@beartype.beartype
def get_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute centroid from bounding box of mask pixels."""
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


@beartype.beartype
def assign_quadrant_ids(
    mask: np.ndarray, img_height: int, img_width: int
) -> np.ndarray:
    """Assign quadrant-based IDs to masks."""
    assert mask.shape == (img_height, img_width), (
        "Mask shape "
        f"{mask.shape} must match image dimensions ({img_height}, {img_width})"
    )
    obj_ids = get_obj_ids(mask)
    assert len(obj_ids) >= 1, "Position mode requires at least 1 mask"
    assert len(obj_ids) <= 4, f"Position mode requires <=4 masks, got {len(obj_ids)}"

    centroids: dict[int, tuple[float, float]] = {}
    for obj_id in obj_ids:
        centroids[obj_id] = get_centroid(mask == obj_id)

    # Compute split point (per-dimension, same-half logic)
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    if len(centroids) == 1:
        # Single mask: use image center
        split_x = image_center_x
        split_y = image_center_y
    else:
        # Multiple masks: per-dimension same-half check
        cx_values = [c[0] for c in centroids.values()]
        cy_values = [c[1] for c in centroids.values()]

        # X dimension: if all on same side of image center, use image center
        all_left = all(cx < image_center_x for cx in cx_values)
        all_right = all(cx > image_center_x for cx in cx_values)
        if all_left or all_right:
            split_x = image_center_x
        else:
            split_x = sum(cx_values) / len(cx_values)

        # Y dimension: if all on same side of image center, use image center
        all_top = all(cy < image_center_y for cy in cy_values)
        all_bottom = all(cy > image_center_y for cy in cy_values)
        if all_top or all_bottom:
            split_y = image_center_y
        else:
            split_y = sum(cy_values) / len(cy_values)

    new_mask = np.zeros_like(mask)
    quadrants_used: set[int] = set()
    eps = 1e-9

    for obj_id, (cx, cy) in centroids.items():
        assert abs(cx - split_x) > eps, (
            f"Centroid x too close to split point: cx={cx}, split_x={split_x}"
        )
        assert abs(cy - split_y) > eps, (
            f"Centroid y too close to split point: cy={cy}, split_y={split_y}"
        )

        if cx < split_x and cy < split_y:
            quadrant_id = 1
        elif cx > split_x and cy < split_y:
            quadrant_id = 2
        elif cx < split_x and cy > split_y:
            quadrant_id = 3
        else:
            quadrant_id = 4

        assert quadrant_id not in quadrants_used, (
            f"Quadrant collision: multiple masks in quadrant {quadrant_id}"
        )
        quadrants_used.add(quadrant_id)

        new_mask[mask == obj_id] = quadrant_id

    return new_mask


@beartype.beartype
def collapse_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse all objects to single ID."""
    return (mask > 0).astype(mask.dtype)


@beartype.beartype
def transform_mask_for_mode(
    mask: np.ndarray, mask_mode: str, img_width: int, img_height: int
) -> np.ndarray:
    """Transform a mask according to the mask mode."""
    if mask_mode == "original":
        return mask
    if mask_mode == "binary":
        return collapse_to_binary(mask)
    if mask_mode == "position":
        return assign_quadrant_ids(mask, img_height, img_width)
    raise ValueError(f"Invalid mask_mode: {mask_mode}")


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


@beartype.beartype
class batched_idx:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """

    def __init__(self, total_size: int, batch_size: int):
        """
        Args:
            total_size: total number of examples
            batch_size: maximum distance between the generated indices
        """
        self.total_size = total_size
        self.batch_size = batch_size

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        """Yield (start, end) index pairs for batching."""
        for start in range(0, self.total_size, self.batch_size):
            stop = min(start + self.batch_size, self.total_size)
            yield start, stop

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.total_size + self.batch_size - 1) // self.batch_size
