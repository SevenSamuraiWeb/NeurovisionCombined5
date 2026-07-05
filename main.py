"""NeuroVision API — FastAPI entrypoint.

Run:  python main.py
  or: venv\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 5000
"""
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.job_state import init_db

Path("temp/jobs").mkdir(parents=True, exist_ok=True)
init_db()

app = FastAPI(title="NeuroVision API")

# flask-cors parity: origins "*" with credentials. allow_origin_regex echoes
# the request Origin (allow_origins=["*"] would not, breaking credentialed
# requests).
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    # Flask envelope parity: 400 {"message": ...} instead of 422 {"detail": ...}
    err = exc.errors()[0] if exc.errors() else {}
    return JSONResponse(status_code=400, content={"message": err.get("msg", "Invalid request")})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    # Dict details (e.g. from utils.uploads.read_image_upload) are the body
    # verbatim, preserving each API's error envelope key.
    content = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


from app.api.jobs import router as jobs_router          # noqa: E402
from app.api.preview import router as preview_router    # noqa: E402
from app.api.storage import router as storage_router    # noqa: E402
from app.identifier.router import router as identifier_router  # noqa: E402
from app.cloak.router import router as cloak_router     # noqa: E402

app.include_router(jobs_router, prefix="/v1/analysis")
app.include_router(preview_router, prefix="/v1/preview")
app.include_router(storage_router, prefix="/v1/storage")
app.include_router(identifier_router, prefix="/identifier")
app.include_router(cloak_router, prefix="/cloak")


@app.get("/")
def root():
    from app.blur.blur_service import loaded as blur_loaded
    from app.classifier.classifier import get_classifier
    from app.fog.fog_service import loaded as fog_loaded
    from app.identifier.identifier_service import _text_embs_loaded
    from app.old.old_service import loaded as old_loaded

    return {
        "message": "NeuroVision API running",
        "models": {
            "classifier": get_classifier.cache_info().currsize > 0,
            "dark": True,        # loaded at import
            "blur": blur_loaded(),
            "fog": fog_loaded(),
            "old_photo": old_loaded(),
            "identifier": _text_embs_loaded,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
