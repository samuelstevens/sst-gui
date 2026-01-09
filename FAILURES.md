# Segmentation Mask Browser - Findings

## RESOLVED: All Masks Were Empty

**Original Finding**: All predicted masks contained only zeros.

### Root Cause

The `processor.add_inputs_to_inference_session()` function in the HuggingFace transformers library stores a reference to the `obj_ids` list passed to it. During the forward pass, it clears this list internally. Since we were passing `all_obj_ids` directly, the original list was cleared, causing the mask combining loop to never execute (iterating over an empty list).

### Fix

In `inference.py:242-244`, pass a **copy** of the list instead of the original:

```python
processor.add_inputs_to_inference_session(
    inference_session=session,
    frame_idx=frame_idx,
    obj_ids=list(all_obj_ids),  # Pass a copy - processor clears this list internally
    input_masks=masks_for_objs,
)
```

### Verification

After the fix, masks now contain proper object IDs:
```
CAM000001_d.JPG.png:
  Unique values: [0 1 2 3 4]
  Non-zero pixels: 24.35%
```

### Note

Existing masks in `pred_masks/` were generated with the buggy code and need to be regenerated to contain valid segmentations.
