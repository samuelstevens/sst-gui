"""Unit tests for mask ID transformation algorithms."""

import numpy as np
import pytest

import lib


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


def test_get_centroid_simple_blob():
    # 10x10 blob at (5,5) to (14,14) -> centroid at (9.5, 9.5)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    cx, cy = lib.get_centroid(mask)
    assert cx == 9.5
    assert cy == 9.5


def test_get_centroid_single_pixel():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 30] = 1
    cx, cy = lib.get_centroid(mask)
    assert cx == 30.0
    assert cy == 50.0


def test_get_centroid_wide_blob():
    # Blob from (10,20) to (90,30) -> centroid at (50, 25)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:31, 10:91] = 1
    cx, cy = lib.get_centroid(mask)
    assert cx == 50.0
    assert cy == 25.0


def test_assign_quadrant_ids_four_masks_clear_quadrants():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    # Check that each original region got the expected quadrant ID
    assert result[10, 10] == 1  # top-left
    assert result[15, 90] == 2  # top-right
    assert result[80, 12] == 3  # bottom-left
    assert result[85, 88] == 4  # bottom-right


def test_assign_quadrant_ids_four_masks_clustered_top_left():
    """4 masks all in top-left region of image use mean split (not image center).

    All 4 centroids are in top-left quadrant of the image, but with 4 masks
    we always use mean, which separates them into 4 distinct quadrants.
    This is the melp_14-108 case that motivated the 4-mask exception.
    """
    # All masks in top-left region of a 100x100 image (image center = 50,50)
    mask = make_mask_with_blobs(
        (100, 100),
        {
            1: (5, 5, 15, 15),  # centroid ~(10, 10)
            2: (25, 8, 35, 18),  # centroid ~(30, 13)
            3: (8, 25, 18, 35),  # centroid ~(13, 30)
            4: (28, 28, 38, 38),  # centroid ~(33, 33)
        },
    )

    result = lib.assign_quadrant_ids(mask, 100, 100)

    # Mean split should be ~(21.5, 21.5), separating into 4 quadrants
    assert result[10, 10] == 1  # top-left of cluster
    assert result[13, 30] == 2  # top-right of cluster
    assert result[30, 13] == 3  # bottom-left of cluster
    assert result[33, 33] == 4  # bottom-right of cluster


def test_assign_quadrant_ids_two_masks_diagonal():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[20, 10] == 1  # top-left -> ID 1
    assert result[70, 80] == 4  # bottom-right -> ID 4


def test_assign_quadrant_ids_two_masks_bottom_half():
    """Two masks both in bottom half use image center for split_y.

    This is the CAM036161_d case that motivated the same-half logic.
    Both masks in bottom half (y > 50), so split_y = 50 (image center).
    Expected: quadrants 3 (bottom-left) and 4 (bottom-right).
    """
    mask = make_mask_with_blobs(
        (100, 100),
        {
            1: (15, 70, 25, 80),  # centroid ~(19.5, 74.5) -> bottom-left
            2: (75, 65, 85, 75),  # centroid ~(79.5, 69.5) -> bottom-right
        },
    )

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[75, 20] == 3  # bottom-left -> ID 3
    assert result[70, 80] == 4  # bottom-right -> ID 4


def test_assign_quadrant_ids_two_masks_horizontal():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    # With mean_y ~50.5, mask 1 (y=50) is top, mask 2 (y=51) is bottom
    # mask 1: x=20 < mean_x=50, y=50 < mean_y=50.5 -> top-left = 1
    # mask 2: x=80 > mean_x=50, y=51 > mean_y=50.5 -> bottom-right = 4
    assert result[50, 20] == 1
    assert result[51, 80] == 4


def test_assign_quadrant_ids_three_masks():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[20, 20] == 1  # top-left
    assert result[20, 80] == 2  # top-right
    assert result[80, 60] == 4  # bottom-right


def test_assign_quadrant_ids_single_mask_top_left():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[20, 20] == 1  # top-left


def test_assign_quadrant_ids_single_mask_bottom_right():
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

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[80, 80] == 4  # bottom-right


def test_assign_quadrant_ids_mask_shape_must_match_img_dimensions():
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
        lib.assign_quadrant_ids(mask, 200, 200)  # Mismatched dimensions


def test_assign_quadrant_ids_single_mask_at_image_center_fails():
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
        lib.assign_quadrant_ids(mask, 100, 100)


def test_assign_quadrant_ids_empty_mask_fails():
    """Empty mask (no objects) should assert - refs must have at least 1 mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(AssertionError, match="Position mode requires at least 1 mask"):
        lib.assign_quadrant_ids(mask, 100, 100)


def test_assign_quadrant_ids_more_than_four_masks_fails():
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
        lib.assign_quadrant_ids(mask, 100, 100)


def test_assign_quadrant_ids_collision_same_quadrant_fails():
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
        lib.assign_quadrant_ids(mask, 100, 100)


def test_assign_quadrant_ids_two_masks_same_x_half_uses_image_center():
    """Two masks in same horizontal half use image center for split_x.

    Two masks with centroids at ~(49.5, 19.5) and ~(49.5, 79.5).
    Both are in left half (x < 50), so split_x = 50 (image center).
    Expected: quadrants 1 (top-left) and 3 (bottom-left).
    """
    mask = make_mask_with_blobs(
        (100, 100),
        {
            1: (45, 15, 55, 25),  # centroid ~(49.5, 19.5) -> top-left
            2: (45, 75, 55, 85),  # centroid ~(49.5, 79.5) -> bottom-left
        },
    )

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[20, 50] == 1  # top-left
    assert result[80, 50] == 3  # bottom-left


def test_assign_quadrant_ids_two_masks_same_y_half_uses_image_center():
    """Two masks in same vertical half use image center for split_y.

    Two masks with centroids at ~(19.5, 49.5) and ~(79.5, 49.5).
    Both are in top half (y < 50), so split_y = 50 (image center).
    Expected: quadrants 1 (top-left) and 2 (top-right).
    """
    mask = make_mask_with_blobs(
        (100, 100),
        {
            1: (15, 45, 25, 55),  # centroid ~(19.5, 49.5) -> top-left
            2: (75, 45, 85, 55),  # centroid ~(79.5, 49.5) -> top-right
        },
    )

    result = lib.assign_quadrant_ids(mask, 100, 100)

    assert result[50, 20] == 1  # top-left
    assert result[50, 80] == 2  # top-right


def test_assign_quadrant_ids_centroid_at_mean_split_point_fails():
    """Centroid at calculated mean split point should fail.

    Two masks spanning both halves: centroids at (40, 50) and (60, 50).
    X: not all_left, not all_right -> split_x = mean = 50
    Y: not all_top, not all_bottom -> split_y = mean = 50
    Both have cy == split_y, should fail.
    """
    mask = make_mask_with_blobs(
        (100, 100),
        {
            # centroid_x = (35+45)/2 = 40, centroid_y = (45+55)/2 = 50
            1: (35, 45, 46, 56),
            # centroid_x = (55+65)/2 = 60, centroid_y = (45+55)/2 = 50
            2: (55, 45, 66, 56),
        },
    )

    with pytest.raises(AssertionError, match="Centroid y too close to split point"):
        lib.assign_quadrant_ids(mask, 100, 100)


def test_get_obj_ids_multiple_objects():
    """Returns sorted list of non-zero object IDs."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            3: (5, 5, 15, 15),
            1: (85, 5, 95, 15),
            4: (5, 85, 15, 95),
        },
    )

    result = lib.get_obj_ids(mask)

    assert result == [1, 3, 4]  # sorted


def test_get_obj_ids_empty_mask():
    """Empty mask returns empty list."""
    mask = np.zeros((100, 100), dtype=np.uint8)

    result = lib.get_obj_ids(mask)

    assert result == []


def test_get_obj_ids_excludes_background():
    """Background (0) is excluded from results."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 5
    mask[30:40, 30:40] = 0  # explicitly background

    result = lib.get_obj_ids(mask)

    assert result == [5]
    assert 0 not in result


def test_transform_mask_for_mode_original_mode_unchanged():
    """Original mode returns mask unchanged."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            7: (5, 5, 15, 15),
            3: (85, 85, 95, 95),
        },
    )

    result = lib.transform_mask_for_mode(mask, "original", 100, 100)

    assert np.array_equal(result, mask)
    assert result[10, 10] == 7
    assert result[90, 90] == 3


def test_transform_mask_for_mode_binary_mode_collapses_ids():
    """Binary mode collapses all non-zero to 1."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            7: (5, 5, 15, 15),
            3: (85, 85, 95, 95),
        },
    )

    result = lib.transform_mask_for_mode(mask, "binary", 100, 100)

    assert result[10, 10] == 1
    assert result[90, 90] == 1
    assert result[50, 50] == 0  # background


def test_transform_mask_for_mode_position_mode_assigns_quadrants():
    """Position mode assigns quadrant-based IDs."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            7: (5, 5, 15, 15),  # top-left -> 1
            3: (85, 85, 95, 95),  # bottom-right -> 4
        },
    )

    result = lib.transform_mask_for_mode(mask, "position", 100, 100)

    assert result[10, 10] == 1  # was 7, now quadrant 1
    assert result[90, 90] == 4  # was 3, now quadrant 4
    assert result[50, 50] == 0  # background


def test_transform_mask_for_mode_invalid_mode_raises_valueerror():
    """Invalid mask_mode raises ValueError."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            1: (5, 5, 15, 15),
        },
    )

    with pytest.raises(ValueError, match="Invalid mask_mode"):
        lib.transform_mask_for_mode(mask, "invalid", 100, 100)


def test_transform_mask_for_mode_position_mode_uses_correct_dimensions():
    """Position mode uses img_width/height for split point calculation."""
    # Create a non-square mask to verify dimensions are used correctly
    mask = np.zeros((200, 100), dtype=np.uint8)  # height=200, width=100
    mask[10:20, 10:20] = 1  # top-left of 200x100 image

    result = lib.transform_mask_for_mode(mask, "position", 100, 200)

    # Centroid ~(14.5, 14.5), image center (50, 100)
    # x=14.5 < 50, y=14.5 < 100 -> top-left = 1
    assert result[15, 15] == 1


def test_collapse_to_binary_multiple_objects():
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

    result = lib.collapse_to_binary(mask)

    assert result[10, 10] == 1
    assert result[10, 90] == 1
    assert result[90, 10] == 1
    assert result[90, 90] == 1
    assert result[50, 50] == 0  # background stays 0


def test_collapse_to_binary_empty_mask():
    """Empty mask stays empty."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = lib.collapse_to_binary(mask)
    assert np.all(result == 0)


def test_collapse_to_binary_single_object():
    """Single object stays as 1."""
    mask = make_mask_with_blobs(
        (100, 100),
        {
            5: (40, 40, 60, 60),
        },
    )

    result = lib.collapse_to_binary(mask)

    assert result[50, 50] == 1
    assert result[10, 10] == 0
