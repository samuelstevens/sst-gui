# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "duckdb",
#     "numpy",
#     "pillow",
#     "pydantic",
#     "term-image",
#     "tyro",
# ]
# ///
"""Preview original images and predicted masks for a list of primary keys."""

import csv
import dataclasses
import importlib
import logging
import pathlib
import sys
import types

import beartype
import duckdb
import numpy as np
import tyro
from PIL import Image

import lib

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("check")

PREFERRED_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".cr2"]
COLORS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ],
    dtype=np.uint8,
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    spec: pathlib.Path
    """Path to spec.json."""

    keys: list[str] = dataclasses.field(default_factory=list)
    """Primary keys to check. If empty, read from stdin."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ResolvedKey:
    raw_key: str
    image_name: str


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImageRecord:
    raw_key: str
    image_name: str
    img_fpath: pathlib.Path
    pred_mask_fpath: pathlib.Path
    ref_mask_fpath: pathlib.Path


@beartype.beartype
def load_spec(spec_fpath: pathlib.Path) -> lib.Spec:
    return lib.Spec.model_validate_json(spec_fpath.read_text())


@beartype.beartype
def read_keys_from_stdin() -> list[str]:
    keys = [line.strip() for line in sys.stdin.read().splitlines()]
    return [key for key in keys if key]


@beartype.beartype
def read_master_image_names(
    master_csv_fpath: pathlib.Path, primary_key: str
) -> list[str]:
    image_names: list[str] = []
    with master_csv_fpath.open(newline="") as fd:
        reader = csv.DictReader(fd)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            msg = f"No header row found in {master_csv_fpath}"
            raise ValueError(msg)
        if primary_key not in fieldnames:
            msg = f"Primary key column '{primary_key}' not found in {master_csv_fpath}"
            raise KeyError(msg)
        for row in reader:
            image_names.append(row[primary_key])
    return image_names


@beartype.beartype
def pick_best_match(matches: list[str]) -> str:
    if len(matches) == 1:
        return matches[0]

    def score(name: str) -> tuple[int, int]:
        ext = pathlib.Path(name.lower()).suffix
        try:
            idx = PREFERRED_EXTS.index(ext)
        except ValueError:
            idx = len(PREFERRED_EXTS)
        return (idx, len(name))

    return sorted(matches, key=score)[0]


@beartype.beartype
def resolve_keys(
    keys: list[str], image_names: list[str]
) -> tuple[list[ResolvedKey], list[str]]:
    resolved: list[ResolvedKey] = []
    missing: list[str] = []
    image_names_lower = [name.lower() for name in image_names]

    for key in keys:
        if key in image_names:
            resolved.append(ResolvedKey(key, key))
            continue

        key_lower = key.lower()
        matches = [
            name
            for name, name_lower in zip(image_names, image_names_lower)
            if name_lower.startswith(key_lower)
        ]
        if matches:
            resolved.append(ResolvedKey(key, pick_best_match(matches)))
            continue

        matches = [
            name
            for name, name_lower in zip(image_names, image_names_lower)
            if key_lower in name_lower
        ]
        if matches:
            resolved.append(ResolvedKey(key, pick_best_match(matches)))
            continue

        missing.append(key)

    return resolved, missing


@beartype.beartype
def lookup_img_paths(
    master_csv_fpath: pathlib.Path,
    spec: lib.Spec,
    image_names: list[str],
) -> dict[str, pathlib.Path]:
    conn = duckdb.connect()
    conn.from_csv_auto(str(master_csv_fpath)).create("master_df")
    conn.execute("CREATE TEMP TABLE keys(pk VARCHAR)")
    conn.executemany("INSERT INTO keys VALUES (?)", [(name,) for name in image_names])

    query = (
        f"SELECT k.pk AS pk, {spec.img_path} AS img_path "
        f"FROM master_df m JOIN keys k ON m.{spec.primary_key} = k.pk"
    )
    rows = conn.execute(query).fetchall()
    return {pk: pathlib.Path(img_path) for pk, img_path in rows}


@beartype.beartype
def build_records(
    spec: lib.Spec, resolved: list[ResolvedKey], img_paths: dict[str, pathlib.Path]
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for item in resolved:
        img_fpath = img_paths.get(item.image_name)
        if img_fpath is None:
            logger.warning("No image path found for %s", item.image_name)
            continue
        pred_mask_fpath = spec.pred_masks / f"{item.image_name}.png"
        ref_mask_fpath = spec.ref_masks / f"{item.image_name}.png"
        records.append(
            ImageRecord(
                item.raw_key,
                item.image_name,
                img_fpath,
                pred_mask_fpath,
                ref_mask_fpath,
            )
        )
    return records


@beartype.beartype
def render_mask_rgb(mask_fpath: pathlib.Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(mask_fpath) as mask:
        mask = mask.convert("L")
        mono = np.array(mask, dtype=np.uint8)

    rgb = np.zeros((mono.shape[0], mono.shape[1], 3), dtype=np.uint8)
    object_vals = [value for value in np.unique(mono) if value > 0]
    for value in object_vals:
        rgb[mono == value] = COLORS[(int(value) - 1) % len(COLORS)]

    mask_rgb = Image.fromarray(rgb)
    if mask_rgb.size != size:
        mask_rgb = mask_rgb.resize(size, Image.Resampling.NEAREST)
    return mask_rgb


@beartype.beartype
def draw_side_by_side(img_fpath: pathlib.Path, mask_fpath: pathlib.Path) -> None:
    with Image.open(img_fpath) as img:
        orientation = read_exif_orientation(img)
        img = apply_exif_orientation(img, orientation).convert("RGB")
        mask_rgb = render_mask_rgb(mask_fpath, img.size)

        gap_px = 10
        combined_w = img.width + gap_px + mask_rgb.width
        combined_h = max(img.height, mask_rgb.height)
        combined = Image.new("RGB", (combined_w, combined_h), (0, 0, 0))
        combined.paste(img, (0, 0))
        combined.paste(mask_rgb, (img.width + gap_px, 0))

    term_image_image: types.ModuleType = importlib.import_module("term_image.image")
    img_obj = term_image_image.AutoImage(combined)
    rendered_w, rendered_h = img_obj.rendered_size
    img_obj.draw(
        h_align="left",
        pad_width=rendered_w,
        v_align="top",
        pad_height=rendered_h,
    )


@beartype.beartype
def read_exif_orientation(img: Image.Image) -> int | None:
    exif = img.getexif()
    if not exif:
        return None
    orientation = exif.get(274)
    if orientation is None:
        return None
    if not isinstance(orientation, int):
        return None
    return orientation


@beartype.beartype
def apply_exif_orientation(img: Image.Image, orientation: int | None) -> Image.Image:
    if orientation is None or orientation == 1:
        return img
    if orientation == 2:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return img.transpose(Image.Transpose.ROTATE_180)
    if orientation == 4:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return img.transpose(Image.Transpose.TRANSPOSE)
    if orientation == 6:
        return img.transpose(Image.Transpose.ROTATE_270)
    if orientation == 7:
        return img.transpose(Image.Transpose.TRANSVERSE)
    if orientation == 8:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


@beartype.beartype
def wait_for_enter() -> bool:
    prompt = "Press Enter for next (q to quit): "
    response = None
    if sys.stdin.isatty():
        response = input(prompt)
    else:
        try:
            with open("/dev/tty") as fd:
                print(prompt, end="", flush=True)
                response = fd.readline()
        except OSError:
            print(prompt)
            return True

    if response is None:
        return True
    if response.strip().lower().startswith("q"):
        return False
    return True


@beartype.beartype
def run(args: Args) -> None:
    spec = load_spec(args.spec)
    keys = args.keys
    if not keys:
        keys = read_keys_from_stdin()
    if not keys:
        logger.error("No keys provided via args or stdin.")
        return

    image_names = read_master_image_names(spec.master_csv, spec.primary_key)
    resolved, missing = resolve_keys(keys, image_names)
    if missing:
        logger.warning("Missing %d keys: %s", len(missing), ", ".join(missing))

    img_paths = lookup_img_paths(
        spec.master_csv, spec, [r.image_name for r in resolved]
    )
    records = build_records(spec, resolved, img_paths)
    if not records:
        logger.error("No matching records found.")
        return

    total = len(records)
    for idx, record in enumerate(records, start=1):
        if not record.img_fpath.exists():
            logger.warning("Image does not exist: %s", record.img_fpath)
            continue
        if record.pred_mask_fpath.exists():
            mask_fpath = record.pred_mask_fpath
            mask_source = "pred"
        elif record.ref_mask_fpath.exists():
            mask_fpath = record.ref_mask_fpath
            mask_source = "ref"
        else:
            logger.warning(
                "Mask missing for %s; pred=%s ref=%s",
                record.image_name,
                record.pred_mask_fpath,
                record.ref_mask_fpath,
            )
            continue
        print(f"[{idx}/{total}] {record.raw_key}")
        print(f"image: {record.img_fpath} | mask ({mask_source}): {mask_fpath}")

        draw_side_by_side(record.img_fpath, mask_fpath)
        if not wait_for_enter():
            return


if __name__ == "__main__":
    run(tyro.cli(Args))
