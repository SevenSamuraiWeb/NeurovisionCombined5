"""Pydantic request models for the JSON-body endpoints."""
from pydantic import BaseModel, ConfigDict


class PipelineStage(BaseModel):
    # Entries also carry "enabled"/"params" (and anything the frontend adds) —
    # preserved verbatim via extra="allow".
    model_config = ConfigDict(extra="allow")
    stage: str


class PipelineUpdateRequest(BaseModel):
    pipeline: list[PipelineStage]


class SaveImageRequest(BaseModel):
    job_id: str
    user_id: str
    image: str  # base64, optionally with a data-URI prefix