# Mask ID Transformation Modes for inference.py

## Overview

This spec describes three mask ID assignment strategies for inference.py. The mode is configured in `spec.json` (created by the GUI) and read by inference.py at runtime.

## Use Cases

- **Wings in different positions**: Butterflies may be oriented differently across images, but wings should have consistent semantic IDs based on their relative position (e.g., top-left wing = ID 1)
- **Color swatches mislabeled**: Reference masks may have inconsistent IDs that need normalization
- **Baseline comparison**: Original mode preserves labeler intent for comparison

## Modes

### Mode 1: `original`

Preserve the labeler's mask IDs exactly as they appear in reference masks. Trust SAM2's object tracking to maintain IDs through inference.

**Behavior:**
- Reference masks: IDs unchanged, passed directly to SAM2
- Predicted masks: Trust SAM2's obj_ids directly (no reassignment)
- No transformation or validation

**Use case:** Baseline comparison, trust labeler's semantic intent and SAM2's tracking.

### Mode 2: `position`

Assign mask IDs based on the relative position of each mask's centroid within a 2x2 quadrant grid.

**Algorithm:**
1. For each reference image, extract all non-background masks
2. Assert at least one mask exists (empty refs are invalid)
3. Compute centroid (bounding box center) for each mask
4. Compute the split point (per-dimension):
   - **4 masks**: Always use mean of centroid values (guarantees each mask gets its own quadrant)
   - **2-3 masks**: Apply same-half logic per dimension:
     - **X dimension**: If all centroids are in the same horizontal half (all left or all right of image center), use `image_width/2`. Otherwise, use mean of centroid x-values.
     - **Y dimension**: If all centroids are in the same vertical half (all above or all below image center), use `image_height/2`. Otherwise, use mean of centroid y-values.
   - **Single mask**: Always use image center (width/2, height/2)
5. Assign quadrant ID based on position relative to split point:
   - `x < split_x` and `y < split_y` → **top-left = 1**
   - `x > split_x` and `y < split_y` → **top-right = 2**
   - `x < split_x` and `y > split_y` → **bottom-left = 3**
   - `x > split_x` and `y > split_y` → **bottom-right = 4**
6. Relabel the reference mask with the computed quadrant ID
7. Pass transformed reference masks to SAM2
8. Trust SAM2's obj_ids for predicted masks (no re-computation)

**Coordinate System:**
- Image coordinates: (0,0) at top-left, y increases downward
- Centroid calculation happens on **original mask at original resolution** (before any resizing to target_size)
- The `img_width` and `img_height` parameters must match `mask.shape`

**Constraints (enforced via assert):**
- `mask.shape` must equal `(img_height, img_width)` - no coordinate space mismatch allowed
- At least 1 mask per reference image (empty refs are invalid)
- Maximum 4 masks per reference image
- No two masks may have the same quadrant (collision)
- Centroid must not be within epsilon (1e-9) of split point on either axis (floating-point safety)

**Sparse Masks:**
- Fewer than 4 masks is allowed
- Quadrant IDs are preserved (e.g., 2 masks might get IDs 1 and 4 if top-left and bottom-right)

**Multiple Reference Frames:**
- Each reference image in a group is transformed independently
- This means the same object may get different quadrant IDs across refs if it moves (intended behavior)
- SAM2 receives the transformed refs and handles object tracking across frames

**Configuration in spec.json:**
```json
{
  "mask_mode": "position"
}
```

No grid_rows/grid_cols needed - always 2x2.

### Mode 3: `binary`

Merge all non-background masks into a single object class before conditioning SAM2.

**Behavior:**
- Reference masks: Collapse all IDs > 0 to ID = 1 **before** passing to SAM2
- SAM2 conditions on a single object
- Predicted masks: Post-process to collapse all IDs > 0 to ID = 1 (ensures consistent output even if SAM2 misbehaves)

**Use case:** Simple foreground/background segmentation, ignoring part labels.

**Configuration in spec.json:**
```json
{
  "mask_mode": "binary"
}
```

## spec.json Schema

```json
{
  "mask_mode": "original" | "position" | "binary"
}
```

**Required fields:**
- `mask_mode`: Must be explicitly set (no default)

## Command Line Interface

```bash
uv run inference.py              # Normal inference
uv run inference.py --dry-run    # Validate without writing output
```

### --dry-run Mode

Validates the configuration and data without running inference or writing output files.

**Reports:**
- Quadrant collisions (position mode): masks in same quadrant
- Missing refs: groups with no reference masks
- Mask counts: number of masks per image, anomalies (>4 in position mode)

## Error Handling

| Condition | Action |
|-----------|--------|
| Empty ref mask (position mode) | Assert failure (refs must have at least 1 mask) |
| Quadrant collision (position mode) | Assert failure (use --dry-run to validate first) |
| More than 4 masks (position mode) | Assert failure |
| Centroid on split point axis (tie) | Assert failure (abs(cx - split_x) < 1e-9 OR abs(cy - split_y) < 1e-9) |
| Invalid mask_mode value | Fail with error |
| Missing refs for group | Log and skip group |

**Skip Behavior:**
- Skipped images are logged with reason
- No sentinel files written - resume logic will retry (user fixes data or uses --dry-run)

## Output

- Output masks follow the same format as input
- Mask IDs in output correspond to the transformed IDs per the selected mode
- No additional metadata files

## Implementation Notes

### Centroid Calculation

```python
def get_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute centroid from bounding box of mask pixels."""
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)
```

### Quadrant Assignment (Position Mode)

```python
def assign_quadrant_ids(mask: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    """Assign quadrant-based IDs to masks.

    Args:
        mask: Single-channel mask with object IDs (0 = background)
        img_height: Original image height (for single-mask case)
        img_width: Original image width (for single-mask case)

    Returns:
        Transformed mask with quadrant-based IDs (1-4)
    """
    assert mask.shape == (img_height, img_width), f"Mask shape {mask.shape} must match ({img_height}, {img_width})"
    obj_ids = [i for i in np.unique(mask) if i > 0]
    assert len(obj_ids) >= 1, "Position mode requires at least 1 mask"
    assert len(obj_ids) <= 4, f"Position mode requires <=4 masks, got {len(obj_ids)}"

    # Compute centroids
    centroids = {}
    for obj_id in obj_ids:
        centroids[obj_id] = get_centroid(mask == obj_id)

    # Compute split point
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    cx_values = [c[0] for c in centroids.values()]
    cy_values = [c[1] for c in centroids.values()]

    if len(centroids) == 1:
        # Single mask: use image center
        split_x = image_center_x
        split_y = image_center_y
    elif len(centroids) == 4:
        # 4 masks: always use mean (guarantees separation)
        split_x = sum(cx_values) / len(cx_values)
        split_y = sum(cy_values) / len(cy_values)
    else:
        # 2-3 masks: per-dimension same-half check
        # X dimension: if all on same side, use image center
        all_left = all(cx < image_center_x for cx in cx_values)
        all_right = all(cx > image_center_x for cx in cx_values)
        if all_left or all_right:
            split_x = image_center_x
        else:
            split_x = sum(cx_values) / len(cx_values)

        # Y dimension: if all on same side, use image center
        all_top = all(cy < image_center_y for cy in cy_values)
        all_bottom = all(cy > image_center_y for cy in cy_values)
        if all_top or all_bottom:
            split_y = image_center_y
        else:
            split_y = sum(cy_values) / len(cy_values)

    # Assign quadrant IDs
    new_mask = np.zeros_like(mask)
    quadrants_used = set()

    EPS = 1e-9
    for obj_id, (cx, cy) in centroids.items():
        assert abs(cx - split_x) > EPS, f"Centroid x too close to split point: cx={cx}, split_x={split_x}"
        assert abs(cy - split_y) > EPS, f"Centroid y too close to split point: cy={cy}, split_y={split_y}"

        if cx < split_x and cy < split_y:
            quadrant_id = 1  # top-left
        elif cx > split_x and cy < split_y:
            quadrant_id = 2  # top-right
        elif cx < split_x and cy > split_y:
            quadrant_id = 3  # bottom-left
        else:
            quadrant_id = 4  # bottom-right

        assert quadrant_id not in quadrants_used, f"Quadrant collision: multiple masks in quadrant {quadrant_id}"
        quadrants_used.add(quadrant_id)

        new_mask[mask == obj_id] = quadrant_id

    return new_mask
```

### Binary Mode Transformation

```python
def collapse_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse all objects to single ID."""
    return (mask > 0).astype(mask.dtype)
```

### Integration Points in inference.py

1. **Load spec.json** at startup, extract `mask_mode`
2. **Transform reference masks** before passing to SAM2:
   - `original`: no-op
   - `position`: apply `assign_quadrant_ids(mask, img_height, img_width)` to each ref mask
   - `binary`: apply `collapse_to_binary()` to each ref mask
3. **Post-process predicted masks:**
   - `original`: no-op
   - `position`: no-op (trust SAM2 IDs)
   - `binary`: apply `collapse_to_binary()` to ensure output is {0, 1}
4. **--dry-run flag**: validate all refs without running inference
