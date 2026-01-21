# Plan: Replace Streamlit GUI

## Goals
- Replace the Streamlit UI in `gui.py` with a faster, more responsive frontend.
- Keep the data flow intact: load CSV metadata, group frames, generate SAM2 masks, pick/label masks, and save `spec.json` + reference masks.
- Minimize regression risk by keeping core logic (mask generation, spec writing) in shared Python modules.
- Favor a single-process Python app if possible.

## Current Understanding
- The GUI is a single Streamlit script (`gui.py`) that:
  - Uploads a CSV, runs a SQL filter via DuckDB, and groups images by user-selected columns.
  - Builds `Frame` objects and samples reference frames.
  - Uses a SAM2 pipeline to generate masks, shows overlays, and saves labeled masks to `root/ref_masks/{pk}.png`.
  - Writes `spec.json` and copies the input CSV to `root/master.csv`.
- The main bottlenecks are Streamlit reruns (toggling a later mask forces full rerender) and image rendering. SAM2 latency is acceptable.
- Desired interaction is list-based selection and fast checkbox ticking; pan/zoom or drawing tools are not required right now.
- Expected scale: 10–20 frames today, likely 100 frames per split, with 8–10 splits.
- Preference is for a single-process Python app, but a frontend + backend split is acceptable if it materially improves responsiveness.

## Migration Options (High-Level)
- Marimo (Python-first notebooks/apps): single-process, reactive DAG, with the ability to isolate expensive work into separate cells and only re-run dependents.
- Elm + Lightstar (SPA frontend + Python backend): strongest UI responsiveness; can ship masks to the browser once and toggle overlays client-side. Requires explicit API design and a two-process setup.
- NiceGUI (Python + FastAPI + Vue/Quasar, websockets): event-driven UI updates without full page reruns; still backend-first, but updates are incremental rather than full rerender.
- Gradio Blocks (Python UI with explicit event handlers/state): event callbacks can target specific outputs; still server round-trips on each interaction.

## Decision
- Proceed with Elm (frontend) + Litestar (backend). The goal is client-side mask toggling with server-side mask generation and persistence.

## Tradeoffs to Evaluate
- Single-process preference pushes toward Marimo or another Python-native UI (e.g., FastHTML + HTMX, NiceGUI, Gradio Blocks), but these must support partial updates and efficient image rendering.
- Two-process setup enables a thin UI that updates independently of heavy Python work, and allows caching/streaming of image overlays without recomputing state. It also allows client-side mask toggling if all masks are shipped to the browser once.
- Given the need for rapid checkbox toggling without reruns, the Elm + Litestar split is likely the best fit if we accept a two-process setup.

## Key Unknowns / Questions
- Which UI frameworks can do partial updates for checkbox toggles without re-rendering the full page, while staying single-process?
- What rendering path is fastest for overlaying masks (matplotlib vs direct image compositing vs canvas vs frontend canvas)?
- Do we need persistent UI state across many frames to avoid reloading images?
- Are we willing to accept a lightweight two-process model if it substantially improves UI responsiveness?
- Clarify whether “Lightstar” refers to Litestar (formerly Starlite) or a different backend framework.

## API Sketch (Elm + Litestar)
- Project lifecycle:
  - `POST /api/projects` (multipart): CSV file + `filter_query`, `group_by`, `img_path`, `primary_key`, `root_dpath`, `sam2_model`, `device` → returns `project_id`, group count, and column metadata.
  - `GET /api/projects/{project_id}`: returns config + dataset summary.
  - `GET /api/projects/{project_id}/spec`: download current `spec.json`.
  - `DELETE /api/projects/{project_id}`: delete cached masks + project state.
- Groups:
  - Group keys are URL-encoded strings (e.g., `species=Heliconius|region=Panama`), or a server-issued group id with a lookup table returned by `GET /groups`.
  - `GET /api/projects/{project_id}/groups?offset&limit`: list group keys and counts.
  - `POST /api/projects/{project_id}/groups/{group_key}/sample`: body `{n_ref_frames, seed}` → list of frames with `pk` + `img_path`.
- Frames:
  - Frames are addressed by `pk` (the CSV primary key) to avoid a separate `frame_id`.
  - `GET /api/projects/{project_id}/frames`: list sampled frames and their cached status.
  - `GET /api/projects/{project_id}/frames/{pk}`: frame metadata.
  - `GET /api/projects/{project_id}/frames/{pk}/image?scale=`: base image preview (scale configurable).
- Masks:
  - `POST /api/projects/{project_id}/frames/{pk}/masks`: idempotent; compute/cache masks if missing, return preview metadata + URLs.
  - `GET /api/projects/{project_id}/frames/{pk}/masks`: fetch cached preview masks; 404 if missing.
- Selection + persistence:
  - `POST /api/projects/{project_id}/frames/{pk}/selection`: body `{mask_ids, labels}` → server writes `ref_masks/{pk}.png` using full-res masks; returns saved path.
  - `PATCH /api/projects/{project_id}/frames/{pk}/selection`: update labels without resending all masks (optional).
  - `DELETE /api/projects/{project_id}/frames/{pk}/selection`: clear saved selection for a frame.
- Health:
  - `GET /health`: liveness endpoint.
- Caching/format:
  - Cache full-res masks on disk (e.g., `root_dpath/cache/masks/{pk}.npz`) and serve downscaled previews for UI.
  - Preview payload: per-mask PNGs at a configurable scale, so Elm can toggle mask visibility client-side.

## Endpoint Semantics (Detailed)
- `POST /api/projects`: create a project session. The server parses the CSV, runs `filter_query`, validates `primary_key` uniqueness, builds grouping metadata, and persists the inputs (like `spec.json` + `master.csv`). Returns `project_id` plus summary data.
- `GET /api/projects/{project_id}`: fetch project summary for reloads, including config, row counts, and group counts.
- `GET /api/projects/{project_id}/spec`: download the current `spec.json` for debugging or downstream inference.
- `GET /api/projects/{project_id}/groups`: list group keys and counts with pagination for large datasets.
- `POST /api/projects/{project_id}/groups/{group_key}/sample`: sample `n_ref_frames` (with `seed`) from a group and return frames with `pk` and `img_path`.
- `GET /api/projects/{project_id}/frames`: list sampled frames and their cached status for reloads.
- `GET /api/projects/{project_id}/frames/{pk}`: return frame metadata and cached status (image size, mask cache presence).
- `GET /api/projects/{project_id}/frames/{pk}/image?scale=`: return a scaled preview image to display as the base layer in Elm.
- `POST /api/projects/{project_id}/frames/{pk}/masks`: run SAM2 if cache is missing, cache full-res masks, and return preview per-mask PNGs at the requested scale.
- `GET /api/projects/{project_id}/frames/{pk}/masks`: fetch cached preview mask PNGs without recompute.
- `POST /api/projects/{project_id}/frames/{pk}/selection`: accept `{mask_ids, labels}` and write `ref_masks/{pk}.png` using full-res cached masks. Returns the saved path.

## Claude Feedback (Summarized)
- Prefer using the CSV primary key (`pk`) in paths instead of a separate `frame_id` to reduce ambiguity.
- Group identifiers should be explicit and stable (e.g., URL-encoded group keys) or have a documented lookup table.
- Make `POST /masks` idempotent and let `GET /masks` 404 when cache is missing, so the client can decide when to compute.
- Add endpoints: `DELETE /api/projects/{project_id}` for cleanup, `GET /api/projects/{project_id}/frames` to list sampled frames after reload, and a `PATCH`/`DELETE` selection endpoint to update or clear labels.
- Consider a simple `GET /health` endpoint and a consistent error schema (`{error, code}`).
- Cache invalidation: include `sam2_model` + preview scale in the cache key; avoid stale masks after config changes.
- Ensure preview masks are cached at least once per scale; avoid recomputing on every request.
- Guard SAM2 calls from blocking the event loop (use worker threads/processes) and add per-frame locks to prevent duplicate compute.
- Persist project state across restarts (disk-backed registry, or rebuild from `spec.json` + caches).
- If frontend/backend are separate origins, configure CORS.

## Gemini Feedback (Summarized)
- Encode group keys safely in URLs (Base64URL or a surrogate ID) to avoid path issues.
- Keep `pk` type consistent across endpoints; apply encoding if it contains special characters.
- Consider `PATCH /api/projects/{project_id}` to update config (sam2 model/device) and trigger cache invalidation.
- Return mask preview URLs in JSON rather than embedding PNG bytes; let the browser fetch and cache them.
- Cache scaled base images as well as masks if resizing is a bottleneck.
- Run SAM2 in a thread/process to avoid blocking the event loop.
- Include model/params hash in cache path to avoid stale masks.
- Add optional batch compute endpoint (e.g., compute masks for all frames in a group).
- Ensure CORS is enabled if Elm is served separately from Litestar.

## Final API Spec (Proposed)
### Conventions
- `project_id`: server-issued opaque id (UUID).
- `pk`: CSV primary key string; if it contains special characters, it must be URL-encoded.
- `group_key`: Base64URL-encoded UTF-8 string of `col=value` pairs joined by `|` (order = `group_by` columns). Example raw key: `species=Heliconius|region=Panama`.
- Error shape: `{ "error": "message", "code": "machine_readable" }`.

### Endpoints (Request/Response Examples)
- `POST /api/projects` (multipart form)  
  **Why keep:** Entry point that parses CSV + config and initializes in-memory state.
  - Fields: `csv` (file), `filter_query` (str), `group_by` (comma-separated), `img_path` (str), `primary_key` (str), `root_dpath` (str), `sam2_model` (str), `device` (str)
  - Response:
```json
{
  "project_id": "6b0fd7b5-6a8f-4c4d-8b8d-96c4c9d4c52b",
  "columns": ["frame_id", "img_path", "species", "region"],
  "group_count": 312,
  "row_count": 50211
}
```
- `GET /api/projects/{project_id}`  
  **Why keep:** Enables browser refresh to restore config + counts without re-uploading.
```json
{
  "project_id": "6b0fd7b5-6a8f-4c4d-8b8d-96c4c9d4c52b",
  "config": {
    "filter_query": "SELECT * FROM master_df",
    "group_by": ["species", "region"],
    "img_path": "img_path",
    "primary_key": "frame_id",
    "sam2_model": "facebook/sam2.1-hiera-tiny",
    "device": "cuda",
    "root_dpath": "/data/sst"
  },
  "row_count": 50211,
  "group_count": 312
}
```
- `GET /api/projects/{project_id}/groups?offset=0&limit=50`  
  **Why keep:** Populate group picker after refresh; can be paginated for large datasets.
- `POST /api/projects/{project_id}/groups/{group_key}/sample`  
  **Why keep:** Core sampling step; can be invoked repeatedly with different seeds.
```json
{
  "groups": [
    {
      "group_key": "c3BlY2llcz1IZWxpY29uaXVzfHJlZ2lvbj1QYW5hbWE",
      "group_display": {"species": "Heliconius", "region": "Panama"},
      "count": 143
    }
  ],
  "total": 312
}
```
- `GET /api/projects/{project_id}/frames`  
  **Why keep:** Allows UI to rebuild selected frame list and cached status after refresh.
```json
{ "n_ref_frames": 5, "seed": 2 }
```
```json
{
  "frames": [
    { "pk": "frame_00017", "img_path": "/data/images/frame_00017.jpg" }
  ]
}
```
- `GET /api/projects/{project_id}/frames/{pk}/image?scale=0.05`  
  **Why keep:** Provides base preview for mask toggling; cacheable by scale.
```json
{
  "frames": [
    { "pk": "frame_00017", "img_path": "/data/images/frame_00017.jpg", "masks_cached": true }
  ]
}
```
- `POST /api/projects/{project_id}/frames/{pk}/masks` (idempotent)  
  **Why keep:** Computes SAM2 masks or returns cached preview URLs.
```json
{ "scale": 0.05 }
```
```json
{
  "scale": 0.05,
  "masks": [
    { "mask_id": 0, "score": 0.92, "area": 18231, "url": "/api/projects/{project_id}/frames/{pk}/masks/0.png?scale=0.05" }
  ]
}
```
- `GET /api/projects/{project_id}/frames/{pk}/masks/{mask_id}.png?scale=0.05`  
  **Why keep:** Individual preview PNGs that the browser can cache and toggle locally.
  - Response: `image/png` per-mask preview (alpha mask).
- `POST /api/projects/{project_id}/frames/{pk}/selection`  
  **Why keep:** Persists labels to `ref_masks/{pk}.png`.
```json
{ "mask_ids": [0, 3, 5], "labels": [1, 2, 2] }
```
```json
{ "saved_fpath": "/data/sst/ref_masks/frame_00017.png" }
```
- `GET /api/projects/{project_id}/spec` → raw `spec.json`  
  **Why keep:** Export required for downstream inference.
- `GET /health` → `{ "status": "ok" }`

### Removed Endpoints (Local-Only, Refresh-Safe)
- `PATCH /api/projects/{project_id}`: config updates; easier to create a new project for local-only runs.
- `DELETE /api/projects/{project_id}`: restarting the server clears all in-memory state.
- `GET /api/projects/{project_id}/frames/{pk}`: frame detail is not required for UI restore.
- `GET /api/projects/{project_id}/frames/{pk}/masks`: redundant with idempotent `POST /masks`.
- `PATCH /api/projects/{project_id}/frames/{pk}/selection` and `DELETE /api/projects/{project_id}/frames/{pk}/selection`: UI can resend full selection each save.

### Caching + Invalidation
- Cache keys include `sam2_model`, `scale`, and any SAM2 parameters. Example:
  - Full-res masks: `root_dpath/cache/masks/{sam2_hash}/{pk}.npz`
  - Preview masks: `root_dpath/cache/previews/{sam2_hash}/{scale}/{pk}/{mask_id}.png`
  - Preview base image: `root_dpath/cache/previews/{scale}/{pk}.png`
- `sam2_hash` can be a short hash of `sam2_model` + parameter JSON. Changing model/params invalidates caches.
- SAM2 inference runs in a worker thread/process; per-`pk` locks prevent duplicate compute.

## Next Steps
- Confirm the API surface (request/response shapes) and mask payload format for client-side toggling.
- Decide on preview scaling strategy (e.g., 5% like current) and whether to always save masks from full-res cached data.
- Draft the Litestar service skeleton and identify which parts of `gui.py` move into shared backend modules.
- Draft the Elm UI flow (project setup → group selection → frame selection → mask toggling → save).
