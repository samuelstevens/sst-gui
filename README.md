# SST-GUI

A GUI for the idea presented in [SST](https://arxiv.org/abs/2501.06749) - segment objects consistently across image groups using SAM2 with reference masks.

## Workflow

1. **app.py** - Start the GUI with `uv run python app.py` and open http://localhost:8000. Upload a CSV with image metadata and configure your project: primary key, image path column/expression, grouping columns, root directory, SAM2 model, and mask mode. For each group, sample reference frames, generate candidate masks, select and label objects with integer IDs, then save to disk.

2. **inference.py** - Run `uv run inference.py spec.json` to propagate reference masks to all images using SAM2's memory attention. Use `--dry-run` to validate without running inference, `--batch-size N` to control batching, or `--max-frames N` to limit frames per group.

3. **check.py** - Validate results with `uv run check.py --spec spec.json KEY1 KEY2 ...` to view images and masks side-by-side in the terminal. Keys can be partial matches (case-insensitive). Use `--pred-only` to skip reference masks. Pipe keys from a file: `cat keys.txt | uv run check.py --spec spec.json --pred-only`.

4. **convert.py** - Convert results to ImgSegFolder format with `uv run convert.py --master-csv path/to/master.csv --pred-masks-dpath path/to/pred_masks --output-dpath path/to/output`. Resizes images (LANCZOS) and masks (NEAREST) to a target size and writes images/training/, annotations/training/, and labels.csv.

5. **upload.py** - Upload to HuggingFace Hub with `uv run upload.py --repo-id username/dataset-name --dataset-dpath path/to/segfolder`. Reads the convert.py output and pushes to the hub.

## Mask Modes

The mask mode controls how mask IDs are assigned during inference:

- `original`: Preserve the labeler's mask IDs exactly. Use when you trust the labeler's semantic intent.
- `position`: Assign IDs based on object position in a 2x2 quadrant grid (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right). Use when objects appear in different orientations across images, like butterfly wings.
- `binary`: Merge all objects to a single foreground class (ID=1). Use for simple foreground/background segmentation.

For position mode, IDs are assigned based on each mask's centroid relative to the mean centroid of all masks in the image.

## Test Cases

Some camera images have EXIF metadata specifying rotation. PIL doesn't auto-apply this rotation, but browsers do, which can cause masks to appear flipped/mirrored relative to the displayed image. The fix is to use `ImageOps.exif_transpose()` when loading images. Test image: `CAM036564_v.JPG` (from dataset=none group, seed 42, frame 0).
