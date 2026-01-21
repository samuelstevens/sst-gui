# Mask Mode Consistency in check.py

## Summary

Ensure `check.py` renders masks using the same `mask_mode` defined in `spec.json`, matching the behavior documented in `docs/002-mask-transform/SPEC.md` and used by `inference.py`. This keeps visual inspection aligned with how inference treats reference and predicted masks.

## Goals

- `check.py` reads `mask_mode` from `spec.json` and applies it when rendering masks.
- Mask transformation logic is shared with `inference.py` to avoid drift.
- Transformation errors are aggregated and reported without stopping the entire check run.

## Non-Goals

- No new CLI flags or overrides for `mask_mode`.
- No changes to the `spec.json` schema.
- No new test cases (existing tests updated to import from lib.py).

## Background

- `spec.json` is parsed by `lib.Spec`; missing `mask_mode` already fails validation.
- `check.py` already requires an explicit spec path (`Args.spec`).
- `inference.py` implements `mask_mode` handling with helper functions (`assign_quadrant_ids`, `collapse_to_binary`, `transform_ref_masks`).

## Requirements

### Mask Mode Behavior (from 002)

- **original**: use mask as-is.
- **binary**: collapse all non-zero pixels to 1.
- **position**: reassign object IDs based on quadrant of each mask’s centroid using the image’s original size.

### Display Logic

- Only transform **reference masks**. Predicted masks are already transformed by inference and should be displayed as-is.
- For `position` mode, use the image size from the mask file dimensions (masks are created after EXIF transpose, so they match the displayed orientation).
- If mask dimensions don't match the EXIF-transposed image dimensions, treat as a transform error.
- Empty reference masks (all zeros) are treated as transform errors.

**EXIF Note**: `inference.py` applies `ImageOps.exif_transpose()` to images before SAM processing, so all masks (ref and pred) are stored in the EXIF-corrected orientation. No additional rotation is needed in `check.py`.

### Error Handling (Aggregate)

- If a mask fails to transform (e.g., assertion failures in `position`, size mismatch), log the error and **skip the entire record** (don't display anything for that image).
- Continue processing remaining records.
- Track the number of transform failures and emit a summary at the end of the run (even if user quits early).

## Implementation Plan

1. **Extract shared transform helpers into `lib.py`**
   - Move or re-home these functions from `inference.py` so both scripts use a single implementation:
     - `get_obj_ids`
     - `get_centroid`
     - `assign_quadrant_ids`
     - `collapse_to_binary`
   - Add a new helper `transform_mask_for_mode(mask, mask_mode, img_width, img_height)`, which:
     - Returns the original mask for `original`.
     - Returns binary-collapsed mask for `binary`.
     - Returns quadrant-assigned mask for `position`.
     - Raises `ValueError` for unknown modes.
   - Note: Use separate `img_width` and `img_height` params (not a tuple) to avoid PIL `(w, h)` vs numpy `(h, w)` confusion.

2. **Update `inference.py` to use `lib.py` helpers**
   - Replace in-file definitions with imports.
   - Keep behavior unchanged.

3. **Update `check.py` mask rendering**
   - Only transform reference masks (when `mask_source == "ref"`). Predicted masks display as-is.
   - Use pre-EXIF image dimensions for `transform_mask_for_mode`.
   - On transform error (including empty masks), skip the record entirely and increment error count.

4. **Update `tests/test_mask_transform.py`**
   - Replace duplicated function implementations with imports from `lib.py`.
   - Existing tests should pass without modification.

5. **Aggregate transform errors**
   - Collect error count during `run` and log a summary after the loop finishes.
   - Use try/finally to ensure summary prints even if user quits early (presses 'q').

## Resolved Questions

- **Ref vs pred transform**: Only transform reference masks. Predicted masks are already correct from inference.
- **EXIF orientation**: Masks are already EXIF-corrected (inference.py transposes images before SAM). No additional rotation needed in check.py.
- **Size mismatch**: Skip record as transform error.
- **Empty ref masks**: Skip record as transform error.
- **Error display**: Skip record entirely (don't show image alone or untransformed mask).
- **Missing mask_mode**: Fails at `lib.Spec` validation (no default).
- **Spec path**: `check.py` requires explicit `--spec` argument.
- **Test updates**: Update imports in test_mask_transform.py to use lib.py functions.
