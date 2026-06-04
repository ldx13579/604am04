from fastapi import APIRouter, BackgroundTasks
from backend.services.online_finetuner import online_finetuner
from backend.database import SessionLocal
from backend.models import FinetuneRun
from backend.schemas import FinetuneStatusResponse, FinetuneRunResponse, FinetuneConfigUpdateRequest
from backend.config import FINETUNE_CONFIG

router = APIRouter(prefix="/api/finetune", tags=["finetune"])


@router.post("/trigger")
async def trigger_finetune(background_tasks: BackgroundTasks):
    if online_finetuner.is_running:
        return {"status": "already_running"}
    background_tasks.add_task(online_finetuner.run_finetune)
    return {"status": "triggered"}


@router.get("/status", response_model=FinetuneStatusResponse)
async def get_status():
    status = online_finetuner.get_status()
    return FinetuneStatusResponse(**status)


@router.get("/runs")
async def list_runs():
    db = SessionLocal()
    try:
        runs = db.query(FinetuneRun).order_by(FinetuneRun.started_at.desc()).limit(50).all()
        return [FinetuneRunResponse.model_validate(r) for r in runs]
    finally:
        db.close()


@router.put("/config")
async def update_config(request: FinetuneConfigUpdateRequest):
    if request.interval_seconds is not None:
        FINETUNE_CONFIG["interval_seconds"] = request.interval_seconds
    if request.min_buffer_size is not None:
        FINETUNE_CONFIG["min_buffer_size"] = request.min_buffer_size
    if request.epochs is not None:
        FINETUNE_CONFIG["epochs"] = request.epochs
    if request.lr is not None:
        FINETUNE_CONFIG["lr"] = request.lr
    return {"status": "updated", "config": FINETUNE_CONFIG}
