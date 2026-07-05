# NeurovisionCombined — Project Context for Claude

## Project Goal

NeuroVision is an image restoration backend. It accepts a degraded image (blurry, dark, foggy, or an old damaged photograph), auto-classifies the degradation type, lets the user confirm or manually override the proposed pipeline, then restores the image with dedicated specialist pipelines while streaming per-stage progress over SSE. It also offers a CLIP-based damage identifier and adversarial image cloaking. All processing runs server-side in **one FastAPI process** (the former blur/fog and old-photo microservices are merged in — no inter-service HTTP).

## Runtime

- **Python 3.12.0** via the project venv: `venv\Scripts\python.exe` (torch 2.12 CPU-only on this machine).
- Start: `python main.py` (or `python -m uvicorn main:app --host 0.0.0.0 --port 5000`).
- basicsr 1.4.2 needs the torchvision `functional_tensor` shim — installed in `app/__init__.py`; never import basicsr before `app`.
- Settings: `config.py` (pydantic-settings, env prefix `NV_`, reads `.env`). All paths are repo-relative; model weights in `models/`, MinIO creds, ChromaDB dir.

## API Surface (all on :5000)

| Method & path | Purpose |
|---|---|
| `GET /` | Health + per-model load status |
| `POST /v1/analysis/jobs` | multipart `image` + `id` → classify, build proposed pipeline (201) |
| `GET /v1/analysis/jobs/{job_id}` | Job status/progress/classification/pipeline |
| `PUT /v1/analysis/jobs/{job_id}/pipeline` | **Manual override** — JSON `{"pipeline": [{stage, enabled, params}]}` |
| `POST /v1/analysis/jobs/{job_id}/execute` | Run pipeline in background thread (202) |
| `GET /v1/analysis/jobs/{job_id}/stream` | SSE: unnamed `data:` frames with JSON `event` key (`pipeline_started`, `stage_started/completed`, `preview_ready`, `restoration_completed`, `save_ready`, `stream_closed`, `job_failed`) |
| `GET /v1/preview/{job_id}/{stage}` | Stage preview image (`final`, `dark`, `damaged_after_lama_inpainting`, …) |
| `POST /v1/storage/save` | JSON `{job_id, user_id, image(base64)}` → MinIO |
| `POST /identifier` (+`/health`) | CLIP zero-shot damage identifier → `{top_label, top_label_short, confidence, all_scores[]}` |
| `POST /cloak` (+`/health`) | Adversarial cloaking (MTCNN+FaceNet + CLIP shield; FGSM/PGD/MI-FGSM) |

Error envelopes: jobs/preview/storage use `{"message": ...}`; identifier/cloak use `{"error": ...}`. Validation errors return **400** (not FastAPI's default 422) via handlers in `main.py`.

## Repository Structure

```
main.py                  # FastAPI app: CORS, error handlers, router registration
config.py                # pydantic-settings (NV_* env overrides)
requirements.txt
models/                  # ALL .pth weights, central (classifier, retinex_gan_8,
                         #   BSRGAN*, deblur_nafnet, old_photo_damage, codeformer,
                         #   scunet, bopbtl_scratch.pt, facelib/)
RAG/                     # single CLIP+ChromaDB package (embedder, db_conn,
                         #   similarity_search, chroma_db/ with 16k embeddings)
utils/
├── minio.py             # upload_to_minio (settings-driven)
└── uploads.py           # shared multipart image validation (one place)
app/
├── __init__.py          # basicsr/torchvision shim
├── core/                # job_state (sqlite), events (in-mem pub/sub),
│   │                    #   pipeline_executor (daemon-thread runner), schemas (Pydantic)
├── api/                 # jobs.py (analysis+pipeline+stream merged), preview.py, storage.py
├── classifier/          # ResNet-50 5-class; get_classifier() lru_cache singleton
├── dark/                # RAGRetinexFormer + RAG references (loads at import)
├── blur/                # deblur.py engine (U-Net+LaMa+NAFNet+BSRGAN), blur_service.de_blur (lazy)
├── fog/                 # fog.py engine (DCP+BSRGAN), fog_service.de_fog (lazy)
├── old/                 # multi-stage old-photo pipeline (lazy), old_service.de_old
├── identifier/          # CLIP zero-shot identifier (router.py + service)
└── cloak/               # adversarial cloaking (router.py + 2 services; eager load)
tests/                   # contract + latency suite (pytest + requests, live server)
temp/jobs/               # per-job artifacts and previews
data/old/pipeline_runs/  # old-photo per-run artifacts (UUID dirs)
```

## Stage services — uniform contract

`de_dark / de_blur / de_fog / de_old (image: PIL, report=None) -> PIL`
`report(substage_name, pil_image)` streams intermediates; the executor saves them as `stage_{stage}_{substage}.jpg` and publishes SSE `preview_ready` events. The canonical stage→service map lives ONLY in `app/core/pipeline_executor.py` (`STAGE_SERVICES`, `STAGE_ORDER`).

Model loading: classifier lazy singleton (`lru_cache`), dark eager at import, blur/fog/old lazy on first use (first such job pays the load), identifier/cloak eager at import (CLIP shared from `RAG.embedder`).

## Tests

```
venv\Scripts\python.exe -m pytest tests/ -m "not slow and not minio" -q   # fast contract
NEW_STACK=1 ... -m pytest tests/ -q -s                                    # full incl. pipelines + latency table
```
`BASE_URL` env targets any live server. Markers: `slow` (full pipeline runs), `minio` (needs MinIO on :9000), `new_only` (merged-backend features).

## Known Issues / Notes

1. **`models/damage_detector.pth` is a mislabeled NAFNet training checkpoint** — the blur pipeline's U-Net scratch detector can't load it and falls back to a heuristic mask (same as the old microservice did). Drop in a real `DamageDetectorUNet` state_dict to enable neural detection.
2. **Old-photo damage classifier segmentation head is OOD-broken** (collapses to damage_ratio ≈ 1.0 on real photos); classical CV mask is primary, model mask corroborating only.
3. Auto-downloaded on first use (network needed once): `RealESRGAN_x4plus.pth`, `codeformer.pth`, `scunet_color_real_psnr.pth`, `ddcolor_modelscope.pth`, facenet MTCNN/VGGFace2, CLIP ViT-B/32.
4. CORS is unrestricted with credentials (`allow_origin_regex=".*"`).
5. Events accumulate in memory per job (`app/core/events.py`) — fine for dev; add cleanup if the process runs for weeks.
6. On this Windows box, `localhost` URLs pay ~2s IPv6-fallback penalty against IPv4-bound servers; use `127.0.0.1` when benchmarking.
7. The React frontend (`C:/project-null`) predates this API redesign and must be updated to it (unprefixed `/analysis`→`/v1/analysis`, `/restore/*` flow replaced by the jobs flow with `damaged_*` sub-stage previews).
