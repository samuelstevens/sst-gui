"""SST-style inference using SAM2 with batched memory attention.

Efficient approach:
1. Process all labeled images through ViT + memory encoder ONCE
2. Build shared memory bank from all labeled frames
3. For batches of unlabeled images, run ViT + memory attention with shared memory

Valid SAM2 model names (HuggingFace):

- facebook/sam2.1-hiera-tiny
- facebook/sam2.1-hiera-small
- facebook/sam2.1-hiera-base-plus
- facebook/sam2.1-hiera-large
"""

import collections
import dataclasses
import logging
import pathlib
import sys
import time

import beartype
import duckdb
import numpy as np
import polars as pl
import torch
import transformers
import tyro
from PIL import Image, ImageOps

import lib

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")


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
        f"Mask shape {mask.shape} must match ({img_height}, {img_width})"
    )
    obj_ids = [int(x) for x in np.unique(mask) if x > 0]
    assert len(obj_ids) >= 1, "Position mode requires at least 1 mask"
    assert len(obj_ids) <= 4, f"Position mode requires <=4 masks, got {len(obj_ids)}"

    centroids: dict[int, tuple[float, float]] = {}
    for obj_id in obj_ids:
        centroids[obj_id] = get_centroid(mask == obj_id)

    if len(centroids) == 1:
        split_x = img_width / 2
        split_y = img_height / 2
    else:
        split_x = sum(c[0] for c in centroids.values()) / len(centroids)
        split_y = sum(c[1] for c in centroids.values()) / len(centroids)

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
@dataclasses.dataclass(frozen=True, slots=True)
class FrameMeta:
    pk: str
    group: tuple
    img_fpath: pathlib.Path
    original_size: tuple[int, int]


@beartype.beartype
def get_frames(spec: lib.Spec) -> list[FrameMeta]:
    """Load and filter frames from master CSV based on spec."""
    master_df = pl.read_csv(spec.master_csv, infer_schema_length=None)
    conn = duckdb.connect()
    conn.register("master_df", master_df)
    filtered_df = conn.execute(spec.filter_query).pl()

    if spec.primary_key not in filtered_df.columns:
        logger.error(
            "Primary key column '%s' not found in filtered dataset", spec.primary_key
        )
        return []

    if spec.img_path not in filtered_df.columns:
        q = f"SELECT *, {spec.img_path} AS img_path FROM filtered_df"
        filtered_df = conn.execute(q).pl()
    else:
        filtered_df = filtered_df.with_columns(pl.col(spec.img_path).alias("img_path"))

    frames: list[FrameMeta] = []
    for row in filtered_df.iter_rows(named=True):
        img_fpath = pathlib.Path(row["img_path"])
        pk = str(row[spec.primary_key])
        group = tuple(row[col] for col in spec.group_by)
        try:
            with Image.open(img_fpath) as im:
                size = im.size
        except FileNotFoundError:
            logger.warning("File '%s' does not exist.", img_fpath)
            continue
        frames.append(FrameMeta(pk, group, img_fpath, size))
    return frames


@beartype.beartype
def get_ref_masks(spec: lib.Spec, group_pks: set[str]) -> dict[str, np.ndarray]:
    """Load reference masks for a group. Returns dict of pk -> mask array."""
    ref_masks: dict[str, np.ndarray] = {}
    for mask_fpath in spec.ref_masks.glob("*.png"):
        pk = mask_fpath.stem
        if pk in group_pks:
            with Image.open(mask_fpath) as m:
                mask = np.array(m, dtype=np.uint8)
            ref_masks[pk] = mask
    return ref_masks


@beartype.beartype
def get_obj_ids(mask: np.ndarray) -> list[int]:
    """Return sorted object IDs in a mask (excluding background)."""
    return [int(x) for x in np.unique(mask) if x > 0]


@beartype.beartype
def transform_ref_masks(
    mask_mode: str,
    ref_frames: list[FrameMeta],
    ref_masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Transform reference masks based on mask_mode."""
    pk_to_frame = {frame.pk: frame for frame in ref_frames}
    transformed: dict[str, np.ndarray] = {}

    for pk, mask in ref_masks.items():
        frame = pk_to_frame.get(pk)
        if frame is None:
            continue

        if mask_mode == "original":
            if not get_obj_ids(mask):
                continue
            transformed[pk] = mask
            continue

        if mask_mode == "binary":
            if not np.any(mask > 0):
                continue
            transformed[pk] = collapse_to_binary(mask)
            continue

        if mask_mode == "position":
            img_width, img_height = frame.original_size
            transformed[pk] = assign_quadrant_ids(mask, img_height, img_width)
            continue

        raise ValueError(f"Invalid mask_mode: {mask_mode}")

    return transformed


class ImageDataset(torch.utils.data.Dataset):
    """Dataset for parallel image loading."""

    def __init__(self, frames: list[FrameMeta], processor):
        self.frames = frames
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        frame = self.frames[idx]
        with Image.open(frame.img_fpath) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        # Process single image - returns [1, 1, C, H, W]
        processed = self.processor.video_processor(
            videos=[img], device="cpu", return_tensors="pt"
        )
        # Return [C, H, W] and pk
        return processed.pixel_values_videos[0, 0], frame.pk


@beartype.beartype
def precompute_vision_features(
    model: transformers.Sam2VideoModel,
    processor: transformers.Sam2VideoProcessor,
    frames: list[FrameMeta],
    device: str,
    batch_size: int,
    num_workers: int = 8,
) -> dict[str, dict]:
    """Pre-compute vision features for all frames using batched inference.

    Returns a dict mapping pk -> {"vision_feats": ..., "vision_pos_embeds": ...}
    """
    logger.info(
        "Pre-computing vision features for %d frames (batch_size=%d, num_workers=%d)...",
        len(frames),
        batch_size,
        num_workers,
    )

    features_cache: dict[str, dict] = {}

    # Use DataLoader for parallel image loading
    dataset = ImageDataset(frames, processor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    processed_count = 0
    for pixel_values, pks in dataloader:
        # pixel_values: [batch, C, H, W], pks: list of primary keys
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pv_batch = pixel_values.to(device, dtype=torch.bfloat16)
            vision_feats, vision_pos_embeds, _, _ = model.get_image_features(pv_batch)

            # Store each frame's features on CPU
            # Features have shape [HW, batch, C] - batch is middle dimension
            for j, pk in enumerate(pks):
                features_cache[pk] = {
                    "vision_feats": [f[:, j : j + 1, :].cpu() for f in vision_feats],
                    "vision_pos_embeds": [
                        p[:, j : j + 1, :].cpu() for p in vision_pos_embeds
                    ],
                }

        processed_count += len(pks)
        if processed_count % (batch_size * 10) == 0 or processed_count >= len(frames):
            logger.info("  Processed %d/%d frames", processed_count, len(frames))

    return features_cache


@beartype.beartype
def predict_with_session(
    model: transformers.Sam2VideoModel,
    processor: transformers.Sam2VideoProcessor,
    target_frames: list[FrameMeta],
    ref_frames: list[FrameMeta],
    ref_masks: dict[str, np.ndarray],
    features_cache: dict[str, dict],
    device: str,
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    """Predict masks for target frames by reusing a single session's memory.

    This builds a session with ref frames, runs them once to populate memory,
    then predicts each target frame using that shared memory.
    """
    # Get all object IDs from reference masks
    all_obj_ids = set()
    for ref_frame in ref_frames:
        mask = ref_masks[ref_frame.pk]
        all_obj_ids.update(get_obj_ids(mask))
    all_obj_ids = sorted(all_obj_ids)

    if not all_obj_ids:
        return [np.zeros((64, 64), dtype=np.uint8) for _ in target_frames]

    # Load reference images
    ref_images = []
    for f in ref_frames:
        with Image.open(f.img_fpath) as img:
            ref_img = img.convert("RGB")
            if ref_img.size != target_size:
                ref_img = ref_img.resize(target_size, Image.Resampling.LANCZOS)
            ref_images.append(ref_img)

    # Create session with ref frames + one placeholder for target
    # We'll swap out the target frame for each prediction
    dummy_target = ref_images[0]  # Use first ref as placeholder
    session_images = ref_images + [dummy_target]

    session = processor.init_video_session(
        video=session_images,
        inference_device=device,
        dtype=torch.bfloat16,
        max_vision_features_cache_size=len(session_images),
    )

    # Inject cached features for ref frames
    for frame_idx, frame in enumerate(ref_frames):
        if frame.pk in features_cache:
            inject_cached_features(session, frame_idx, features_cache[frame.pk], device)

    # Add mask inputs on reference frames
    n_refs = len(ref_frames)
    for frame_idx, ref_frame in enumerate(ref_frames):
        mask = ref_masks[ref_frame.pk]
        mask_h, mask_w = mask.shape
        if (mask_w, mask_h) != target_size:
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
            mask = np.array(mask_img)
        masks_for_objs = []
        for obj_id in all_obj_ids:
            obj_mask = (mask == obj_id).astype(np.float32)
            masks_for_objs.append(torch.from_numpy(obj_mask))
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=list(
                all_obj_ids
            ),  # Pass a copy - processor clears this list internally
            input_masks=masks_for_objs,
        )

    # Run forward on reference frames to establish conditioning
    logger.info("Running forward on %d reference frames to build memory...", n_refs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame_idx in range(n_refs):
            model(inference_session=session, frame_idx=frame_idx)

    # Now predict each target frame
    results = []
    for target_frame in target_frames:
        # Inject target's vision features
        if target_frame.pk in features_cache:
            inject_cached_features(
                session, n_refs, features_cache[target_frame.pk], device
            )

        # Clear any previous non-cond output at this frame index
        for obj_idx in range(len(all_obj_ids)):
            if n_refs in session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"]:
                del session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"][
                    n_refs
                ]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inference_session=session, frame_idx=n_refs)

        # Post-process mask
        pred_masks = processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[[session.video_height, session.video_width]],
            binarize=True,
        )[0]

        # Combine masks into single channel
        # Use session.obj_idx_to_id to get correct ID for each output index
        # (model may reorder objects internally)
        n_obj, _, h, w = pred_masks.shape
        combined_mask = torch.zeros((h, w), dtype=torch.uint8, device=pred_masks.device)
        for obj_idx in range(n_obj):
            obj_id = session.obj_idx_to_id(obj_idx)
            combined_mask[pred_masks[obj_idx, 0]] = obj_id

        results.append(combined_mask.cpu().numpy())

    return results


@beartype.beartype
def inject_cached_features(
    session,
    frame_idx: int,
    cached: dict,
    device: str,
):
    """Inject pre-computed vision features into session cache."""
    vision_feats = cached["vision_feats"]
    vision_pos_embeds = cached["vision_pos_embeds"]

    session.cache._vision_features[frame_idx] = {
        "vision_feats": [f.to(device) for f in vision_feats],
        "vision_pos_embeds": [p.to(device) for p in vision_pos_embeds],
    }


@beartype.beartype
def run_dry_run(spec: lib.Spec, frame_groups: dict[tuple, list[FrameMeta]]) -> None:
    """Validate reference masks without running inference."""
    logger.info("Running dry-run validation (mask_mode=%s)", spec.mask_mode)
    error_count = 0

    for group, group_frames in frame_groups.items():
        group_pks = set(frame.pk for frame in group_frames)
        ref_masks = get_ref_masks(spec, group_pks)
        if not ref_masks:
            logger.warning("Skipping group %s: no reference masks.", group)
            continue

        ref_pks = set(ref_masks.keys())
        ref_frames = [frame for frame in group_frames if frame.pk in ref_pks]
        for ref_frame in ref_frames:
            mask = ref_masks[ref_frame.pk]
            obj_ids = get_obj_ids(mask)
            logger.info("Group %s ref %s: %d masks", group, ref_frame.pk, len(obj_ids))
            if not obj_ids:
                logger.warning(
                    "Group %s ref %s has no labeled masks.", group, ref_frame.pk
                )

        try:
            transformed = transform_ref_masks(spec.mask_mode, ref_frames, ref_masks)
        except (AssertionError, ValueError) as exc:
            logger.error("Group %s validation error: %s", group, exc)
            error_count += 1
            continue

        if not transformed:
            logger.warning("Skipping group %s: no usable reference masks.", group)
            continue

    if error_count:
        logger.error("Dry run completed with %d validation errors.", error_count)
    else:
        logger.info("Dry run completed with no validation errors.")


@beartype.beartype
def main(
    spec_fpath: pathlib.Path,
    /,
    batch_size: int = 16,
    max_frames: int | None = None,
    dry_run: bool = False,
):
    """Run batched SAM2 inference with shared memory bank.

    Args:
        spec_fpath: Path to spec JSON file
        batch_size: Batch size for inference (number of target frames per batch)
        max_frames: Maximum number of frames per group (for debugging). If None, process all.
        dry_run: Validate inputs without running inference or writing outputs.
    """
    spec = lib.Spec.model_validate_json(spec_fpath.read_text())

    frames = get_frames(spec)
    logger.info("Loaded %d frames", len(frames))
    if not frames:
        logger.error("No frames found")
        return

    valid_modes = {"original", "position", "binary"}
    if spec.mask_mode not in valid_modes:
        logger.error("Invalid mask_mode: %s", spec.mask_mode)
        return

    # Group frames
    frame_groups: dict[tuple, list[FrameMeta]] = collections.defaultdict(list)
    for frame in frames:
        frame_groups[frame.group].append(frame)

    if dry_run:
        run_dry_run(spec, frame_groups)
        return

    spec.pred_masks.mkdir(parents=True, exist_ok=True)

    # Filter to groups with reference masks
    # group_key (columns) -> (list of frames, reference masks [pk -> mask array])
    groups_to_process: dict[tuple, tuple[list[FrameMeta], dict[str, np.ndarray]]] = {}
    for group, group_frames in frame_groups.items():
        group_pks = set(frame.pk for frame in group_frames)
        raw_ref_masks = get_ref_masks(spec, group_pks)
        if not raw_ref_masks:
            logger.warning("Skipping group %s: no reference masks.", group)
            continue

        raw_ref_pks = set(raw_ref_masks.keys())
        raw_ref_frames = [frame for frame in group_frames if frame.pk in raw_ref_pks]
        ref_masks = transform_ref_masks(spec.mask_mode, raw_ref_frames, raw_ref_masks)
        if not ref_masks:
            logger.warning("Skipping group %s: no usable reference masks.", group)
            continue

        ref_pks = set(ref_masks.keys())
        ref_frames_list = [frame for frame in group_frames if frame.pk in ref_pks]

        # Limit frames per group for debugging (keeping all reference frames)
        if max_frames is not None and len(group_frames) > max_frames:
            other_frames = [frame for frame in group_frames if frame.pk not in ref_pks]
            n_other = max(0, max_frames - len(ref_frames_list))
            group_frames = ref_frames_list + other_frames[:n_other]
            logger.info(
                "Group %s limited to %d frames (%d refs + %d targets)",
                group,
                len(group_frames),
                len(ref_frames_list),
                n_other,
            )

        groups_to_process[group] = (group_frames, ref_masks)

    if not groups_to_process:
        logger.error("No groups have reference masks. Cannot proceed.")
        return

    logger.info("Processing %d groups with reference masks.", len(groups_to_process))

    # Load model
    logger.info("Loading model: %s", spec.sam2)
    model = transformers.Sam2VideoModel.from_pretrained(spec.sam2).to(
        spec.device, dtype=torch.bfloat16
    )
    processor = transformers.Sam2VideoProcessor.from_pretrained(spec.sam2)
    model.eval()

    # Count total targets
    total_targets = sum(
        len(group_frames) - len(ref_masks)
        for group_frames, ref_masks in groups_to_process.values()
    )
    total_processed = 0
    start_time = time.time()

    # Process each group
    for group, (group_frames, ref_masks) in groups_to_process.items():
        ref_pks = set(ref_masks.keys())
        ref_frames = [f for f in group_frames if f.pk in ref_pks]
        target_frames = [f for f in group_frames if f.pk not in ref_pks]

        # Filter out already processed targets
        targets_to_process = []
        for tf in target_frames:
            out_fpath = spec.pred_masks / f"{tf.pk}.png"
            if not out_fpath.exists():
                targets_to_process.append(tf)
            else:
                total_processed += 1

        if not targets_to_process:
            logger.info("Group %s: all targets already processed, skipping.", group)
            continue

        logger.info(
            "Processing group %s: %d ref frames, %d targets to process",
            group,
            len(ref_frames),
            len(targets_to_process),
        )

        # Get a common target size (use first target's size)
        target_size = targets_to_process[0].original_size

        # Pre-compute vision features for ref frames first (always needed)
        ref_features_cache = precompute_vision_features(
            model, processor, ref_frames, spec.device, batch_size=batch_size
        )

        # Process targets in batches to avoid OOM
        vision_batch_size = 2048
        for batch_start in range(0, len(targets_to_process), vision_batch_size):
            batch_end = min(batch_start + vision_batch_size, len(targets_to_process))
            batch_targets = targets_to_process[batch_start:batch_end]

            logger.info(
                "Processing batch %d-%d of %d targets",
                batch_start,
                batch_end,
                len(targets_to_process),
            )

            # Pre-compute vision features for this batch of targets
            batch_features_cache = precompute_vision_features(
                model, processor, batch_targets, spec.device, batch_size=batch_size
            )

            # Combine ref features with batch features
            features_cache = {**ref_features_cache, **batch_features_cache}

            # Process targets using session-based approach
            try:
                pred_masks = predict_with_session(
                    model,
                    processor,
                    batch_targets,
                    ref_frames,
                    ref_masks,
                    features_cache,
                    spec.device,
                    target_size,
                )

                # Save each mask
                for target_frame, pred_mask in zip(batch_targets, pred_masks):
                    out_fpath = spec.pred_masks / f"{target_frame.pk}.png"
                    if spec.mask_mode == "binary":
                        pred_mask = collapse_to_binary(pred_mask)
                    mask_img = Image.fromarray(pred_mask)
                    mask_img = mask_img.resize(
                        target_frame.original_size, Image.Resampling.NEAREST
                    )
                    mask_img.save(out_fpath)
                    total_processed += 1

                    # Progress logging every 100 frames
                    if total_processed % 100 == 0 or total_processed == total_targets:
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        eta = (
                            (total_targets - total_processed) / rate if rate > 0 else 0
                        )
                        logger.info(
                            "Progress: %d/%d (%.1f%%) | %.2f frames/sec | ETA: %.1f min",
                            total_processed,
                            total_targets,
                            100 * total_processed / total_targets
                            if total_targets > 0
                            else 100,
                            rate,
                            eta / 60,
                        )

            except Exception as e:
                import traceback

                logger.error(
                    "Error processing group %s batch %d-%d: %s\n%s",
                    group,
                    batch_start,
                    batch_end,
                    e,
                    traceback.format_exc(),
                )
                continue

            # Clear batch cache to free memory
            del batch_features_cache
            del features_cache

        # Clear ref features cache
        del ref_features_cache

    elapsed = time.time() - start_time
    logger.info(
        "Done! Processed %d frames in %.1f seconds (%.2f frames/sec)",
        total_processed,
        elapsed,
        total_processed / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    tyro.cli(main)
