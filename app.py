import base64
import functools
import io
import pathlib
import random
import threading
import uuid
from dataclasses import dataclass, field

import beartype
import duckdb
import numpy as np
import polars as pl
import torch
from litestar import Litestar, Request, Response, get, post
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from litestar.static_files import create_static_files_router
from PIL import Image
from pydantic import BaseModel
from transformers import pipeline


class ProjectConfig(BaseModel):
    filter_query: str
    group_by: list[str]
    img_path: str
    primary_key: str
    root_dpath: str
    sam2_model: str
    device: str


class ProjectSummary(BaseModel):
    project_id: str
    columns: list[str]
    group_count: int
    row_count: int
    config: ProjectConfig | None = None


class GroupSummary(BaseModel):
    group_key: str
    group_display: dict[str, str]
    count: int


class FrameSummary(BaseModel):
    pk: str
    img_path: str
    masks_cached: bool = False


class MaskMeta(BaseModel):
    mask_id: int
    score: float | None = None
    area: int | None = None
    url: str


class MasksResponse(BaseModel):
    scale: float
    masks: list[MaskMeta]


class SelectionRequest(BaseModel):
    mask_ids: list[int]
    labels: list[int]


class ErrorResponse(BaseModel):
    error: str
    code: str


@dataclass(slots=True)
class FrameRecord:
    pk: str
    img_path: str


@dataclass(slots=True)
class ProjectState:
    config: ProjectConfig
    columns: list[str]
    row_count: int
    group_count: int
    groups: list[GroupSummary]
    frames_by_group: dict[str, list[FrameRecord]]
    sampled_frames: dict[str, list[FrameRecord]] = field(default_factory=dict)


_PROJECTS: dict[str, ProjectState] = {}
_MASK_PREVIEWS: dict[tuple[str, str, float], list[bytes]] = {}
_FULL_MASKS: dict[tuple[str, str], list[np.ndarray]] = {}  # full-res masks for saving
_BASE_PREVIEWS: dict[tuple[str, str, float], bytes] = {}
_MASK_COMPUTE_LOCK = threading.Lock()  # Prevents duplicate mask computation

DEFAULT_MASK_SCALE = 0.25
dist_dpath = pathlib.Path(__file__).parent / "dist"
index_fpath = pathlib.Path(__file__).parent / "web" / "index.html"


@functools.lru_cache(maxsize=1)
def load_sam2_pipeline(model_name: str, device: str):
    """Load SAM2 pipeline (cached)."""
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return pipeline(
        "mask-generation",
        model=model_name,
        device=device,
        points_per_batch=256,
    )


@beartype.beartype
def generate_masks_sync(
    sam2_model: str, device: str, img_fpath: pathlib.Path, scale: float
) -> tuple[list[np.ndarray], list[bytes], list[float], list[int]]:
    """Run SAM2 inference synchronously (called from thread pool).

    Returns (full_masks, preview_pngs, scores, areas).
    """
    generator = load_sam2_pipeline(sam2_model, device)

    with Image.open(img_fpath) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        outputs = generator(img, points_per_batch=256)

    full_masks: list[np.ndarray] = []
    preview_pngs: list[bytes] = []
    scores: list[float] = []
    areas: list[int] = []

    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    for item in outputs["masks"]:
        mask_arr = np.array(item, dtype=np.uint8)
        full_masks.append(mask_arr)
        scores.append(float(item.get("score", 0.0)) if hasattr(item, "get") else 0.0)
        areas.append(int(mask_arr.sum()))
        mask_img = Image.fromarray(mask_arr * 255)
        mask_scaled = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        mask_scaled.save(buf, format="PNG")
        preview_pngs.append(buf.getvalue())

    return full_masks, preview_pngs, scores, areas


@beartype.beartype
def encode_group_key(raw_key: str) -> str:
    raw_bytes = raw_key.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw_bytes).decode("ascii")
    return encoded.rstrip("=")


@beartype.beartype
def get_project_or_404(project_id: str) -> ProjectState:
    project = _PROJECTS.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


@beartype.beartype
def error_response(message: str, code: str, *, status_code: int) -> Response:
    payload = ErrorResponse(error=message, code=code).model_dump()
    return Response(content=payload, status_code=status_code)


@beartype.beartype
def build_group_key(group_cols: list[str], row: dict[str, str]) -> str:
    parts = [f"{col}={row[col]}" for col in group_cols]
    return encode_group_key("|".join(parts))


@beartype.beartype
def get_frame(project: ProjectState, pk: str) -> FrameRecord | None:
    return next(
        (
            f
            for frames in project.frames_by_group.values()
            for f in frames
            if f.pk == pk
        ),
        None,
    )


@post("/api/projects")
@beartype.beartype
async def create_project(request: Request) -> Response:
    form = await request.form()
    if "csv" not in form:
        return error_response("missing csv upload", "missing_csv", status_code=400)

    csv_file = form.get("csv")
    if not isinstance(csv_file, UploadFile):
        return error_response("invalid csv upload", "invalid_csv", status_code=400)

    csv_bytes = await csv_file.read()
    df = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=None)

    filter_query = str(form.get("filter_query", "SELECT * FROM master_df"))
    group_by_raw = str(form.get("group_by", "")).strip()
    img_path = str(form.get("img_path", "")).strip()
    primary_key = str(form.get("primary_key", "")).strip()
    root_dpath = str(form.get("root_dpath", "")).strip()
    sam2_model = str(form.get("sam2_model", "facebook/sam2.1-hiera-tiny")).strip()
    device = str(form.get("device", "")).strip()

    if not group_by_raw:
        return error_response(
            "group_by is required", "missing_group_by", status_code=400
        )
    if not img_path:
        return error_response(
            "img_path is required", "missing_img_path", status_code=400
        )
    if not primary_key:
        return error_response(
            "primary_key is required", "missing_primary_key", status_code=400
        )

    group_cols = [col.strip() for col in group_by_raw.split(",") if col.strip()]

    conn = duckdb.connect()
    conn.register("master_df", df)
    filtered_df = conn.execute(filter_query).pl()

    if primary_key not in filtered_df.columns:
        return error_response(
            "primary_key not found in filtered data", "bad_primary_key", status_code=400
        )

    duplicate_count = (
        filtered_df.group_by(primary_key).agg(pl.len()).filter(pl.col("len") > 1)
    )
    if len(duplicate_count) > 0:
        return error_response(
            "primary_key is not unique", "primary_key_not_unique", status_code=400
        )

    if img_path not in filtered_df.columns:
        conn.register("filtered_df", filtered_df)
        path_query = f"SELECT *, {img_path} as img_path FROM filtered_df"
        filtered_df = conn.execute(path_query).pl()
    else:
        filtered_df = filtered_df.with_columns(pl.col(img_path).alias("img_path"))

    grouped = (
        filtered_df.group_by(group_cols)
        .agg([pl.col("img_path"), pl.col(primary_key)])
        .sort(by=group_cols)
    )

    groups: list[GroupSummary] = []
    frames_by_group: dict[str, list[FrameRecord]] = {}
    for row in grouped.iter_rows(named=True):
        group_display = {col: str(row[col]) for col in group_cols}
        group_key = build_group_key(group_cols, group_display)
        img_paths = [str(path) for path in row["img_path"]]
        pks = [str(pk) for pk in row[primary_key]]
        frames = [
            FrameRecord(pk=pk, img_path=img_path)
            for pk, img_path in zip(pks, img_paths)
        ]
        frames_by_group[group_key] = frames
        groups.append(
            GroupSummary(
                group_key=group_key, group_display=group_display, count=len(frames)
            )
        )

    config = ProjectConfig(
        filter_query=filter_query,
        group_by=group_cols,
        img_path=img_path,
        primary_key=primary_key,
        root_dpath=root_dpath,
        sam2_model=sam2_model,
        device=device,
    )

    project_id = str(uuid.uuid4())
    _PROJECTS[project_id] = ProjectState(
        config=config,
        columns=list(df.columns),
        row_count=len(filtered_df),
        group_count=len(groups),
        groups=groups,
        frames_by_group=frames_by_group,
    )

    summary = ProjectSummary(
        project_id=project_id,
        columns=list(df.columns),
        row_count=len(filtered_df),
        group_count=len(groups),
    )
    return Response(content=summary.model_dump(), status_code=201)


@get("/api/projects/{project_id:str}")
@beartype.beartype
async def get_project(project_id: str) -> ProjectSummary:
    project = get_project_or_404(project_id)
    return ProjectSummary(
        project_id=project_id,
        columns=project.columns,
        row_count=project.row_count,
        group_count=project.group_count,
        config=project.config,
    )


@get("/api/projects/{project_id:str}/groups")
@beartype.beartype
async def list_groups(
    project_id: str, offset: int = 0, limit: int = 50
) -> dict[str, object]:
    project = get_project_or_404(project_id)
    groups = project.groups[offset : offset + limit]
    return {
        "groups": [group.model_dump() for group in groups],
        "total": project.group_count,
    }


@post("/api/projects/{project_id:str}/groups/{group_key:str}/sample")
@beartype.beartype
async def sample_group(
    project_id: str, group_key: str, request: Request
) -> dict[str, object]:
    project = get_project_or_404(project_id)
    if group_key not in project.frames_by_group:
        raise HTTPException(status_code=404, detail="group not found")

    payload = await request.json()
    n_ref_frames = int(payload.get("n_ref_frames", 0))
    seed = int(payload.get("seed", 0))
    if n_ref_frames <= 0:
        raise HTTPException(status_code=400, detail="n_ref_frames must be positive")

    frames = project.frames_by_group[group_key]
    if n_ref_frames > len(frames):
        n_ref_frames = len(frames)

    rng = random.Random(seed)
    sample = rng.sample(frames, k=n_ref_frames)
    project.sampled_frames[group_key] = sample

    return {
        "frames": [
            FrameSummary(pk=frame.pk, img_path=frame.img_path).model_dump()
            for frame in sample
        ]
    }


@get("/api/projects/{project_id:str}/frames")
@beartype.beartype
async def list_frames(project_id: str) -> dict[str, object]:
    project = get_project_or_404(project_id)
    frames = []
    for group_frames in project.sampled_frames.values():
        for frame in group_frames:
            cached = (project_id, frame.pk) in _FULL_MASKS
            frames.append(
                FrameSummary(pk=frame.pk, img_path=frame.img_path, masks_cached=cached)
            )
    return {"frames": [frame.model_dump() for frame in frames]}


@get("/api/projects/{project_id:str}/frames/{pk:str}/image")
@beartype.beartype
async def get_frame_image(project_id: str, pk: str, scale: float = 0.25) -> Response:
    project = get_project_or_404(project_id)
    frame = get_frame(project, pk)
    if frame is None:
        raise HTTPException(status_code=404, detail="frame not found")

    cache_key = (project_id, pk, scale)
    cached = _BASE_PREVIEWS.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    img_fpath = pathlib.Path(frame.img_path)
    if not img_fpath.exists():
        raise HTTPException(status_code=404, detail="image not found")

    with Image.open(img_fpath) as img:
        img = img.convert("RGB")
        w, h = img.size
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = buffer.getvalue()
        _BASE_PREVIEWS[cache_key] = data
        return Response(content=data, media_type="image/png")


@beartype.beartype
def build_masks_response(
    project_id: str,
    pk: str,
    scale: float,
    n_masks: int,
    scores: list[float],
    areas: list[int],
) -> MasksResponse:
    return MasksResponse(
        scale=scale,
        masks=[
            MaskMeta(
                mask_id=i,
                score=scores[i] if i < len(scores) else None,
                area=areas[i] if i < len(areas) else None,
                url=f"/api/projects/{project_id}/frames/{pk}/masks/{i}?scale={scale}",
            )
            for i in range(n_masks)
        ],
    )


class ComputeMasksRequest(BaseModel):
    scale: float = 0.25


@post("/api/projects/{project_id:str}/frames/{pk:str}/masks", sync_to_thread=True)
@beartype.beartype
def compute_masks(project_id: str, pk: str, data: ComputeMasksRequest) -> Response:
    project = get_project_or_404(project_id)
    scale = data.scale
    cache_key = (project_id, pk, scale)

    # Return cached if available
    if cache_key in _MASK_PREVIEWS:
        previews = _MASK_PREVIEWS[cache_key]
        response = build_masks_response(project_id, pk, scale, len(previews), [], [])
        return Response(content=response.model_dump(), status_code=200)

    with _MASK_COMPUTE_LOCK:
        # Double-check after acquiring lock
        if cache_key in _MASK_PREVIEWS:
            previews = _MASK_PREVIEWS[cache_key]
            response = build_masks_response(
                project_id, pk, scale, len(previews), [], []
            )
            return Response(content=response.model_dump(), status_code=200)

        frame = get_frame(project, pk)
        if frame is None:
            raise HTTPException(status_code=404, detail="frame not found")

        img_fpath = pathlib.Path(frame.img_path)
        if not img_fpath.exists():
            raise HTTPException(status_code=404, detail="image not found")

        full_masks, preview_pngs, scores, areas = generate_masks_sync(
            project.config.sam2_model,
            project.config.device,
            img_fpath,
            scale,
        )

        _FULL_MASKS[(project_id, pk)] = full_masks
        _MASK_PREVIEWS[cache_key] = preview_pngs

        response = build_masks_response(
            project_id, pk, scale, len(preview_pngs), scores, areas
        )
        return Response(content=response.model_dump(), status_code=200)


@get("/api/projects/{project_id:str}/frames/{pk:str}/masks/{mask_id:int}")
@beartype.beartype
async def get_mask_preview(
    project_id: str, pk: str, mask_id: int, scale: float = 0.25
) -> Response:
    cache_key = (project_id, pk, scale)
    masks = _MASK_PREVIEWS.get(cache_key)
    if masks is None:
        return error_response(
            "mask cache missing", "mask_cache_missing", status_code=404
        )
    if mask_id < 0 or mask_id >= len(masks):
        return error_response(
            "mask_id out of range", "mask_id_out_of_range", status_code=404
        )
    return Response(content=masks[mask_id], media_type="image/png")


@post("/api/projects/{project_id:str}/frames/{pk:str}/selection")
@beartype.beartype
async def save_selection(project_id: str, pk: str, data: SelectionRequest) -> Response:
    project = get_project_or_404(project_id)
    if not data.mask_ids:
        return error_response(
            "mask_ids is required", "missing_mask_ids", status_code=400
        )
    if len(data.mask_ids) != len(data.labels):
        return error_response(
            "mask_ids and labels length mismatch", "length_mismatch", status_code=400
        )

    full_masks = _FULL_MASKS.get((project_id, pk))
    if full_masks is None:
        return error_response(
            "masks not computed yet", "masks_not_computed", status_code=400
        )

    for mask_id in data.mask_ids:
        if mask_id < 0 or mask_id >= len(full_masks):
            return error_response(
                f"mask_id {mask_id} out of range",
                "mask_id_out_of_range",
                status_code=400,
            )

    h, w = full_masks[0].shape
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask_id, label in zip(data.mask_ids, data.labels):
        mask = full_masks[mask_id]
        combined[mask > 0] = label

    root_dpath = pathlib.Path(project.config.root_dpath)
    ref_masks_dpath = root_dpath / "ref_masks"
    ref_masks_dpath.mkdir(parents=True, exist_ok=True)
    saved_fpath = ref_masks_dpath / f"{pk}.png"
    Image.fromarray(combined).save(saved_fpath)

    return Response(content={"saved_fpath": str(saved_fpath)}, status_code=200)


@get("/api/projects/{project_id:str}/spec")
@beartype.beartype
async def get_spec(project_id: str) -> Response:
    project = get_project_or_404(project_id)
    return Response(content=project.config.model_dump(), status_code=200)


@get("/health")
@beartype.beartype
async def health() -> dict[str, str]:
    return {"status": "ok"}


@get("/")
@beartype.beartype
async def index() -> Response:
    if not index_fpath.exists():
        return Response(
            content=b"index.html not found",
            media_type="text/plain",
            status_code=404,
        )
    return Response(content=index_fpath.read_bytes(), media_type="text/html")


def create_app() -> Litestar:
    dist_router = create_static_files_router(
        path="/dist",
        directories=[dist_dpath],
        html_mode=False,
        name="dist",
        include_in_schema=False,
    )
    return Litestar(
        route_handlers=[
            create_project,
            get_project,
            list_groups,
            sample_group,
            list_frames,
            get_frame_image,
            compute_masks,
            get_mask_preview,
            save_selection,
            get_spec,
            health,
            index,
            dist_router,
        ],
        request_max_body_size=100 * 1024 * 1024,  # 100MB
    )


app = create_app()
