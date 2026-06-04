import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from backend.algorithms.trainer import trainer
from backend.schemas import TrainingStartRequest, TrainingRunResponse
from backend.database import SessionLocal
from backend.models import TrainingRun
from typing import List

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingRunResponse)
async def start_training(request: TrainingStartRequest, background_tasks: BackgroundTasks):
    if request.algorithm not in ("cql", "cql_rnn", "dqn", "behavior_cloning", "ensemble_cql"):
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

    db = SessionLocal()
    try:
        from backend.config import DEFAULT_HYPERPARAMS
        merged = {**DEFAULT_HYPERPARAMS.get(request.algorithm, {}), **(request.hyperparameters or {})}

        run = TrainingRun(
            algorithm=request.algorithm,
            hyperparameters=merged,
            status="pending",
            total_epochs=merged.get("epochs", 200),
            current_epoch=0,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id
        run_response = TrainingRunResponse.model_validate(run)
    finally:
        db.close()

    background_tasks.add_task(trainer.start_training, request.algorithm, request.hyperparameters, run_id)
    return run_response


@router.get("/runs", response_model=List[TrainingRunResponse])
async def list_runs():
    db = SessionLocal()
    try:
        runs = db.query(TrainingRun).order_by(TrainingRun.id.desc()).all()
        return [TrainingRunResponse.model_validate(r) for r in runs]
    finally:
        db.close()


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
async def get_run(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return TrainingRunResponse.model_validate(run)
    finally:
        db.close()


@router.delete("/runs/{run_id}")
async def delete_run(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        db.delete(run)
        db.commit()
        return {"message": f"Run {run_id} deleted"}
    finally:
        db.close()
