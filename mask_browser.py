import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo
    import numpy as np
    import polars as pl
    from PIL import Image

    return Image, mo, np, pathlib, pl


@app.cell
def _(pathlib):
    root = pathlib.Path("/local/scratch/stevens.994/datasets/cambridge-segmented")
    pred_masks_dir = root / "pred_masks"
    ref_masks_dir = root / "ref_masks"
    return pred_masks_dir, root


@app.cell
def _(pl, root):
    master_df = pl.read_csv(root / "master.csv", infer_schema_length=None)
    master_df.head()
    return (master_df,)


@app.cell
def _(master_df, mo):
    datasets = sorted(master_df["Dataset"].drop_nulls().unique().to_list())
    dataset_dropdown = mo.ui.dropdown(
        options=["All", "(No Dataset)"] + datasets,
        value="All",
        label="Dataset",
    )
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(dataset_dropdown, master_df, pl, pred_masks_dir):
    def get_img_path(filepath: str) -> str:
        return "/local/scratch/datasets/jiggins/butterflies" + filepath[6:]

    if dataset_dropdown.value == "All":
        filtered_df = master_df
    elif dataset_dropdown.value == "(No Dataset)":
        filtered_df = master_df.filter(pl.col("Dataset").is_null())
    else:
        filtered_df = master_df.filter(pl.col("Dataset") == dataset_dropdown.value)

    filtered_df = filtered_df.with_columns(
        pl.col("filepath")
        .map_elements(get_img_path, return_dtype=pl.Utf8)
        .alias("img_path")
    )

    frames = []
    for _row in filtered_df.iter_rows(named=True):
        pk = _row["Image_name"]
        mask_path = pred_masks_dir / f"{pk}.png"
        if mask_path.exists():
            frames.append({
                "pk": pk,
                "img_path": _row["img_path"],
                "mask_path": str(mask_path),
                "dataset": _row["Dataset"],
            })

    all_frames_df = pl.DataFrame(frames)
    return (all_frames_df,)


@app.cell
def _(mo):
    n_objects_dropdown = mo.ui.dropdown(
        options=["All", "1", "2", "3", "4", "5", "6+"],
        value="All",
        label="# Objects",
    )
    n_objects_dropdown
    return (n_objects_dropdown,)


@app.cell
def _(Image, all_frames_df, mo, n_objects_dropdown, np, pl):
    def count_objects(mask_path: str) -> int:
        """Count non-zero unique values in mask (number of objects)."""
        with Image.open(mask_path) as m:
            arr = np.array(m)
        return len([x for x in np.unique(arr) if x > 0])

    if n_objects_dropdown.value == "All":
        frames_df = all_frames_df
    else:
        # Only count objects when filtering, stop after 32 matches
        target = (
            6 if n_objects_dropdown.value == "6+" else int(n_objects_dropdown.value)
        )
        is_gte = n_objects_dropdown.value == "6+"

        matching_frames = []
        with mo.status.progress_bar(
            total=len(all_frames_df), title="Scanning masks..."
        ) as bar:
            for _row in all_frames_df.iter_rows(named=True):
                n_obj = count_objects(_row["mask_path"])
                if (is_gte and n_obj >= target) or (not is_gte and n_obj == target):
                    matching_frames.append({**_row, "n_objects": n_obj})
                    if len(matching_frames) >= 32:
                        break
                bar.update()

        frames_df = pl.DataFrame(matching_frames)

    frames_df
    return (frames_df,)


@app.cell
def _(frames_df, mo):
    mo.md(f"""
    Found **{len(frames_df)}** images with predicted masks.
    """)
    return


@app.cell
def _(mo):
    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _(frames_df, mo, set_i):
    next_button = mo.ui.button(
        label="Next", on_change=lambda _: set_i(lambda v: (v + 1) % len(frames_df))
    )
    prev_button = mo.ui.button(
        label="Previous", on_change=lambda _: set_i(lambda v: (v - 1) % len(frames_df))
    )
    return next_button, prev_button


@app.cell
def _(frames_df, get_i, mo, set_i):
    idx_slider = mo.ui.slider(
        0,
        max(0, len(frames_df) - 1),
        value=get_i(),
        on_change=lambda i: set_i(i),
        full_width=True,
        label="Index",
    )
    return (idx_slider,)


@app.cell
def _(frames_df, get_i, mo):
    pk_search = mo.ui.text(
        value=frames_df["pk"][get_i()] if len(frames_df) > 0 else "",
        label="Search PK",
    )
    search_button = mo.ui.run_button(label="Go")
    return pk_search, search_button


@app.cell
def _(frames_df, mo, pk_search, pl, search_button, set_i):
    mo.stop(not search_button.value)

    matches = frames_df.filter(pl.col("pk").str.contains(pk_search.value))
    if len(matches) > 0:
        match_idx = frames_df.with_row_index().filter(pl.col("pk") == matches["pk"][0])[
            "index"
        ][0]
        set_i(match_idx)
        search_result = mo.callout(f"Found: **{matches['pk'][0]}**", kind="success")
    else:
        search_result = mo.callout(
            f"No match found for '{pk_search.value}'", kind="warn"
        )
    search_result
    return


@app.cell
def _(mo):
    show_original = mo.ui.switch(value=True, label="Original Image")
    show_mask = mo.ui.switch(value=True, label="Predicted Mask")
    show_overlay = mo.ui.switch(value=True, label="Overlay")
    cols_slider = mo.ui.slider(1, 6, value=3, label="Columns")
    return cols_slider, show_mask, show_original, show_overlay


@app.cell
def _(
    cols_slider,
    frames_df,
    get_i,
    idx_slider,
    mo,
    next_button,
    pk_search,
    prev_button,
    search_button,
    show_mask,
    show_original,
    show_overlay,
):
    current_frame = frames_df.row(get_i(), named=True) if len(frames_df) > 0 else None
    if current_frame:
        dataset_label = current_frame["dataset"] or "(No Dataset)"
        info_str = f"**{current_frame['pk']}** ({get_i() + 1}/{len(frames_df)}) | Dataset: {dataset_label}"
    else:
        info_str = "No frames"

    mo.vstack([
        mo.hstack([prev_button, next_button, mo.md(info_str)], justify="start"),
        mo.hstack([idx_slider], justify="start"),
        mo.hstack([pk_search, search_button], justify="start"),
        mo.hstack(
            [
                show_original,
                show_mask,
                show_overlay,
                mo.md(f"Cols: {cols_slider.value}"),
                cols_slider,
            ],
            justify="start",
        ),
    ])
    return (current_frame,)


@app.cell
def _(Image, np):
    # Target ~200KB: for JPEG at quality 85, ~800px max dimension is reasonable
    MAX_DISPLAY_SIZE = 800

    def resize_for_display(img: Image.Image) -> Image.Image:
        """Resize image to fit within MAX_DISPLAY_SIZE while preserving aspect ratio."""
        w, h = img.size
        if max(w, h) <= MAX_DISPLAY_SIZE:
            return img
        scale = MAX_DISPLAY_SIZE / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def create_overlay(
        img_path: str, mask_path: str, alpha: float = 0.5
    ) -> Image.Image:
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)

        overlay = np.array(img).copy()
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        unique_ids = [x for x in np.unique(mask_arr) if x > 0]
        for i, obj_id in enumerate(unique_ids):
            color = colors[i % len(colors)]
            mask_region = mask_arr == obj_id
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_region,
                    (1 - alpha) * overlay[:, :, c] + alpha * color[c],
                    overlay[:, :, c],
                )

        result = Image.fromarray(overlay.astype(np.uint8))
        return resize_for_display(result)

    def colorize_mask(mask_path: str) -> Image.Image:
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)

        colors = [
            (0, 0, 0),  # 0 = black (background)
            (255, 0, 0),  # 1 = red
            (0, 255, 0),  # 2 = green
            (0, 0, 255),  # 3 = blue
            (255, 255, 0),  # 4 = yellow
            (255, 0, 255),  # 5 = magenta
            (0, 255, 255),  # 6 = cyan
            (255, 128, 0),  # 7 = orange
            (128, 0, 255),  # 8 = purple
            (0, 255, 128),  # 9 = spring green
        ]

        colored = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
        for obj_id in np.unique(mask_arr):
            if obj_id < len(colors):
                color = colors[obj_id]
            else:
                color = colors[obj_id % len(colors)]
            mask_region = mask_arr == obj_id
            colored[mask_region] = color

        result = Image.fromarray(colored)
        return resize_for_display(result)

    return colorize_mask, create_overlay, resize_for_display


@app.cell
def _(
    Image,
    colorize_mask,
    create_overlay,
    current_frame,
    mo,
    pathlib,
    resize_for_display,
    show_mask,
    show_original,
    show_overlay,
):
    def display_frame(frame: dict):
        if frame is None:
            return mo.md("No frame to display")

        imgs = []
        n_views = sum([show_original.value, show_mask.value, show_overlay.value])
        if n_views == 0:
            return mo.md("Enable at least one view")

        img_path = frame["img_path"]
        mask_path = frame["mask_path"]

        if not pathlib.Path(img_path).exists():
            return mo.md(f"Image not found: {img_path}")

        if show_original.value:
            orig_img = Image.open(img_path).convert("RGB")
            orig_img = resize_for_display(orig_img)
            imgs.append(mo.image(orig_img, rounded=True))

        if show_mask.value:
            colored_mask = colorize_mask(mask_path)
            imgs.append(mo.image(colored_mask, rounded=True))

        if show_overlay.value:
            overlay = create_overlay(img_path, mask_path)
            imgs.append(mo.image(overlay, rounded=True))

        return mo.hstack(imgs, widths="equal")

    display_frame(current_frame)
    return (display_frame,)


@app.cell
def _(
    cols_slider,
    display_frame,
    frames_df,
    get_i,
    mo,
    show_mask,
    show_original,
    show_overlay,
):
    n_cols = cols_slider.value
    n_display = min(3, len(frames_df) - get_i())

    mo.stop(
        not any([show_original.value, show_mask.value, show_overlay.value]),
        mo.md("Enable at least one view above."),
    )

    rows = []
    for row_start in range(0, n_display, n_cols):
        row_end = min(row_start + n_cols, n_display)
        row_items = []
        for j in range(row_start, row_end):
            _idx = get_i() + j
            if _idx < len(frames_df):
                frame = frames_df.row(_idx, named=True)
                row_items.append(
                    mo.vstack(
                        [
                            mo.md(f"**{frame['pk']}**"),
                            display_frame(frame),
                        ],
                        align="center",
                    )
                )
        if row_items:
            rows.append(mo.hstack(row_items, widths="equal"))

    mo.vstack(rows)
    return


if __name__ == "__main__":
    app.run()
