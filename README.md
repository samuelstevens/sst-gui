# SST-GUI

A GUI for the idea presented in [SST](https://arxiv.org/abs/2501.06749) - segment objects consistently across image groups using SAM2 with reference masks.

## Workflow

Start the GUI with `uv run python app.py` and open http://localhost:8000 in your browser.

Upload a CSV with image metadata and configure your project. You'll specify a primary key (unique identifier for each image), an image path column or expression, columns to group images by (e.g., by specimen or session), and a root directory for resolving image paths. You'll also choose a SAM2 model variant and a mask mode for inference.

For each group, sample some reference frames and generate candidate masks with SAM2. Select and label the objects you care about by assigning integer IDs, then save the masks to disk.

Run inference with `uv run inference.py spec.json`. This propagates your reference masks to all other images in each group using SAM2's memory attention mechanism. Use `--dry-run` to validate reference masks without running inference, `--batch-size N` to control batch size, or `--max-frames N` to limit frames per group for debugging.

Validate results with `uv run view.py path/to/pred_masks/image.png`.

## Mask Modes

The mask mode controls how mask IDs are assigned during inference:

- `original`: Preserve the labeler's mask IDs exactly. Use this when you trust the labeler's semantic intent.
- `position`: Assign IDs based on object position in a 2x2 quadrant grid (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right). Use this when objects appear in different orientations across images, like butterfly wings.
- `binary`: Merge all objects to a single foreground class (ID=1). Use this for simple foreground/background segmentation.

For position mode, IDs are assigned based on each mask's centroid relative to the mean centroid of all masks in the image.

## Screenshots

First, upload metadata and tell the system how to get your images.

![Upload a CSV file.](docs/assets/metadata.png)

![Configure how to read and group images](docs/assets/configure.png)

SAM2 generates candidate masks for sampled reference frames:

![All masks in an image](docs/assets/all-masks.png)

Filter and label the objects you care about:

![Filtered masks in an image](docs/assets/filtered-masks.png)

View predicted masks:

![View mask](docs/assets/example-mask.png)

## Test Cases

Some camera images have EXIF metadata specifying rotation. PIL doesn't auto-apply this rotation, but browsers do, which can cause masks to appear flipped/mirrored relative to the displayed image. The fix is to use `ImageOps.exif_transpose()` when loading images. Test image: `CAM036564_v.JPG` (from dataset=none group, seed 42, frame 0).
