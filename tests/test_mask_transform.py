"""Unit tests for mask ID transformation algorithms."""

import numpy as np
import pytest


def get_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Compute centroid from bounding box of mask pixels."""
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)


def assign_quadrant_ids(
    mask: np.ndarray, img_height: int, img_width: int
) -> np.ndarray:
    """Assign quadrant-based IDs to masks.

    Args:
        mask: Single-channel mask with object IDs (0 = background)
        img_height: Original image height (for single-mask case)
        img_width: Original image width (for single-mask case)

    Returns:
        Transformed mask with quadrant-based IDs (1-4)
    """
    assert mask.shape == (img_height, img_width), (
        f"Mask shape {mask.shape} must match image dimensions ({img_height}, {img_width})"
    )
    obj_ids = [i for i in np.unique(mask) if i > 0]
    assert len(obj_ids) >= 1, "Position mode requires at least 1 mask"
    assert len(obj_ids) <= 4, f"Position mode requires <=4 masks, got {len(obj_ids)}"

    # Compute centroids
    centroids = {}
    for obj_id in obj_ids:
        centroids[obj_id] = get_centroid(mask == obj_id)

    # Compute split point
    if len(centroids) == 1:
        # Single mask: use image center
        split_x = img_width / 2
        split_y = img_height / 2
    else:
        # Multiple masks: use mean of centroids
        split_x = sum(c[0] for c in centroids.values()) / len(centroids)
        split_y = sum(c[1] for c in centroids.values()) / len(centroids)

    # Assign quadrant IDs
    new_mask = np.zeros_like(mask)
    quadrants_used = set()
    EPS = 1e-9

    for obj_id, (cx, cy) in centroids.items():
        assert abs(cx - split_x) > EPS, (
            f"Centroid x too close to split point: cx={cx}, split_x={split_x}"
        )
        assert abs(cy - split_y) > EPS, (
            f"Centroid y too close to split point: cy={cy}, split_y={split_y}"
        )

        if cx < split_x and cy < split_y:
            quadrant_id = 1  # top-left
        elif cx > split_x and cy < split_y:
            quadrant_id = 2  # top-right
        elif cx < split_x and cy > split_y:
            quadrant_id = 3  # bottom-left
        else:
            quadrant_id = 4  # bottom-right

        assert quadrant_id not in quadrants_used, (
            f"Quadrant collision: multiple masks in quadrant {quadrant_id}"
        )
        quadrants_used.add(quadrant_id)

        new_mask[mask == obj_id] = quadrant_id

    return new_mask


def collapse_to_binary(mask: np.ndarray) -> np.ndarray:
    """Collapse all objects to single ID."""
    return (mask > 0).astype(mask.dtype)


def make_mask_with_blobs(
    size: tuple[int, int], blobs: dict[int, tuple[int, int, int, int]]
) -> np.ndarray:
    """Create a mask with rectangular blobs at specified locations.

    Args:
        size: (height, width) of the mask
        blobs: dict mapping obj_id to (x_min, y_min, x_max, y_max)

    Returns:
        Single-channel mask with object IDs
    """
    mask = np.zeros(size, dtype=np.uint8)
    for obj_id, (x_min, y_min, x_max, y_max) in blobs.items():
        mask[y_min:y_max, x_min:x_max] = obj_id
    return mask


class TestGetCentroid:
    def test_simple_blob(self):
        # 10x10 blob at (5,5) to (14,14) -> centroid at (9.5, 9.5)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        cx, cy = get_centroid(mask)
        assert cx == 9.5
        assert cy == 9.5

    def test_single_pixel(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 30] = 1
        cx, cy = get_centroid(mask)
        assert cx == 30.0
        assert cy == 50.0

    def test_wide_blob(self):
        # Blob from (10,20) to (90,30) -> centroid at (50, 25)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:31, 10:91] = 1
        cx, cy = get_centroid(mask)
        assert cx == 50.0
        assert cy == 25.0


class TestAssignQuadrantIds:
    def test_four_masks_clear_quadrants(self):
        """4 masks in clear quadrants from interview example.

        Centroids: A(10,10), B(90,15), C(12,80), D(88,85)
        Mean: (50, 47.5)
        Expected: A=1 (top-left), B=2 (top-right), C=3 (bottom-left), D=4 (bottom-right)
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (5, 5, 15, 15),  # centroid ~(10, 10) -> top-left
                2: (85, 10, 95, 20),  # centroid ~(90, 15) -> top-right
                3: (7, 75, 17, 85),  # centroid ~(12, 80) -> bottom-left
                4: (83, 80, 93, 90),  # centroid ~(88, 85) -> bottom-right
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        # Check that each original region got the expected quadrant ID
        assert result[10, 10] == 1  # top-left
        assert result[15, 90] == 2  # top-right
        assert result[80, 12] == 3  # bottom-left
        assert result[85, 88] == 4  # bottom-right

    def test_two_masks_diagonal(self):
        """2 masks from interview example.

        Centroids: A(10,20), B(80,70)
        Mean: (45, 45)
        Expected: A=1 (top-left), B=4 (bottom-right)
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (5, 15, 15, 25),  # centroid ~(10, 20) -> top-left
                2: (75, 65, 85, 75),  # centroid ~(80, 70) -> bottom-right
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        assert result[20, 10] == 1  # top-left -> ID 1
        assert result[70, 80] == 4  # bottom-right -> ID 4

    def test_two_masks_horizontal(self):
        """2 masks side by side (left/right).

        Centroids: A(20, 50), B(80, 51)
        Mean: (50, 50.5)
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (15, 45, 25, 55),  # centroid ~(20, 50)
                2: (75, 46, 85, 56),  # centroid ~(80, 51) - slightly below mean
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        # With mean_y ~50.5, mask 1 (y=50) is top, mask 2 (y=51) is bottom
        # mask 1: x=20 < mean_x=50, y=50 < mean_y=50.5 -> top-left = 1
        # mask 2: x=80 > mean_x=50, y=51 > mean_y=50.5 -> bottom-right = 4
        assert result[50, 20] == 1
        assert result[51, 80] == 4

    def test_three_masks(self):
        """3 masks - one quadrant empty.

        Centroids: A(20,20), B(80,20), C(60,80)
        Mean: (53.33, 40)
        Expected: A=1 (top-left), B=2 (top-right), C=4 (bottom-right)
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (15, 15, 25, 25),  # centroid ~(20, 20) -> top-left
                2: (75, 15, 85, 25),  # centroid ~(80, 20) -> top-right
                3: (55, 75, 65, 85),  # centroid ~(60, 80) -> bottom-right (x > mean)
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        assert result[20, 20] == 1  # top-left
        assert result[20, 80] == 2  # top-right
        assert result[80, 60] == 4  # bottom-right

    def test_single_mask_top_left(self):
        """Single mask in top-left quadrant of image.

        With one mask, split point = image center (50, 50).
        Mask centroid ~(20, 20) -> top-left = 1
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (15, 15, 25, 25),  # centroid ~(20, 20)
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        assert result[20, 20] == 1  # top-left

    def test_single_mask_bottom_right(self):
        """Single mask in bottom-right quadrant of image.

        With one mask, split point = image center (50, 50).
        Mask centroid ~(80, 80) -> bottom-right = 4
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (75, 75, 85, 85),  # centroid ~(80, 80)
            },
        )

        result = assign_quadrant_ids(mask, 100, 100)

        assert result[80, 80] == 4  # bottom-right

    def test_mask_shape_must_match_img_dimensions(self):
        """Mask shape must equal (img_height, img_width)."""
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (75, 75, 85, 85),
            },
        )

        with pytest.raises(
            AssertionError, match="Mask shape .* must match image dimensions"
        ):
            assign_quadrant_ids(mask, 200, 200)  # Mismatched dimensions

    def test_single_mask_at_image_center_fails(self):
        """Single mask centered on image center should fail (tie on both axes)."""
        # Blob from (45,45) to (55,55) exclusive -> indices 45-54
        # Bounding box: x_min=45, x_max=54, y_min=45, y_max=54
        # Centroid = ((45+54)/2, (45+54)/2) = (49.5, 49.5)
        # Need centroid at exactly (50, 50) = image center
        # For centroid_x = 50: (x_min + x_max) / 2 = 50 -> x_min + x_max = 100
        # Use x_min=45, x_max=55 -> slice is 45:56
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (45, 45, 56, 56),  # centroid = ((45+55)/2, (45+55)/2) = (50, 50)
            },
        )

        with pytest.raises(AssertionError, match="Centroid x too close to split point"):
            assign_quadrant_ids(mask, 100, 100)

    def test_empty_mask_fails(self):
        """Empty mask (no objects) should assert - refs must have at least 1 mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(
            AssertionError, match="Position mode requires at least 1 mask"
        ):
            assign_quadrant_ids(mask, 100, 100)

    def test_more_than_four_masks_fails(self):
        """More than 4 masks should assert."""
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (5, 5, 15, 15),
                2: (85, 5, 95, 15),
                3: (5, 85, 15, 95),
                4: (85, 85, 95, 95),
                5: (45, 45, 55, 55),
            },
        )

        with pytest.raises(AssertionError, match="Position mode requires <=4 masks"):
            assign_quadrant_ids(mask, 100, 100)

    def test_collision_same_quadrant_fails(self):
        """Two masks in the same quadrant should assert.

        Centroids: A(10,10), B(20,15), C(12,80), D(88,85)
        Mean_x = (10+20+12+88)/4 = 32.5
        Mean_y = (10+15+80+85)/4 = 47.5
        A: x=10 < 32.5, y=10 < 47.5 -> top-left
        B: x=20 < 32.5, y=15 < 47.5 -> top-left (COLLISION!)
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (5, 5, 15, 15),  # centroid ~(10, 10)
                2: (15, 10, 25, 20),  # centroid ~(20, 15)
                3: (7, 75, 17, 85),  # centroid ~(12, 80)
                4: (83, 80, 93, 90),  # centroid ~(88, 85)
            },
        )

        with pytest.raises(AssertionError, match="Quadrant collision"):
            assign_quadrant_ids(mask, 100, 100)

    def test_centroid_on_x_axis_fails(self):
        """Centroid with cx == split_x should fail even if cy != split_y.

        Two masks with centroids at (50, 20) and (50, 80).
        Mean = (50, 50). Both have cx == split_x, should fail.
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (45, 15, 55, 25),  # centroid = (50, 20)
                2: (45, 75, 55, 85),  # centroid = (50, 80)
            },
        )

        with pytest.raises(AssertionError, match="Centroid x too close to split point"):
            assign_quadrant_ids(mask, 100, 100)

    def test_centroid_on_y_axis_fails(self):
        """Centroid with cy == split_y should fail even if cx != split_x.

        Two masks with centroids at (20, 50) and (80, 50).
        Mean = (50, 50). Both have cy == split_y, should fail.
        """
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (15, 45, 25, 55),  # centroid = (20, 50)
                2: (75, 45, 85, 55),  # centroid = (80, 50)
            },
        )

        with pytest.raises(AssertionError, match="Centroid y too close to split point"):
            assign_quadrant_ids(mask, 100, 100)


class TestCollapseToBinary:
    def test_multiple_objects(self):
        """Multiple object IDs collapse to 1."""
        mask = make_mask_with_blobs(
            (100, 100),
            {
                1: (5, 5, 15, 15),
                2: (85, 5, 95, 15),
                3: (5, 85, 15, 95),
                4: (85, 85, 95, 95),
            },
        )

        result = collapse_to_binary(mask)

        assert result[10, 10] == 1
        assert result[10, 90] == 1
        assert result[90, 10] == 1
        assert result[90, 90] == 1
        assert result[50, 50] == 0  # background stays 0

    def test_empty_mask(self):
        """Empty mask stays empty."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = collapse_to_binary(mask)
        assert np.all(result == 0)

    def test_single_object(self):
        """Single object stays as 1."""
        mask = make_mask_with_blobs(
            (100, 100),
            {
                5: (40, 40, 60, 60),
            },
        )

        result = collapse_to_binary(mask)

        assert result[50, 50] == 1
        assert result[10, 10] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
