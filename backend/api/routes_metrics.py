from fastapi import APIRouter, HTTPException, Query
from backend.schemas import MetricResponse, MetricsListResponse
from backend.database import SessionLocal
from backend.models import TrainingRun, TrainingMetric
from backend.algorithms.trainer import trainer
from typing import List

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/runs/{run_id}", response_model=MetricsListResponse)
async def get_run_metrics(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        metrics = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.run_id == run_id)
            .order_by(TrainingMetric.epoch)
            .all()
        )
        return MetricsListResponse(
            run_id=run_id,
            algorithm=run.algorithm,
            metrics=[MetricResponse.model_validate(m) for m in metrics],
        )
    finally:
        db.close()


@router.get("/compare")
async def compare_runs(run_ids: str = Query(..., description="Comma-separated run IDs")):
    ids = [int(x.strip()) for x in run_ids.split(",")]
    db = SessionLocal()
    try:
        results = []
        for rid in ids:
            run = db.query(TrainingRun).get(rid)
            if not run:
                continue
            metrics = (
                db.query(TrainingMetric)
                .filter(TrainingMetric.run_id == rid)
                .order_by(TrainingMetric.epoch)
                .all()
            )
            results.append({
                "run_id": rid,
                "algorithm": run.algorithm,
                "hyperparameters": run.hyperparameters,
                "best_reward": run.best_reward,
                "metrics": [MetricResponse.model_validate(m).model_dump() for m in metrics],
            })
        return results
    finally:
        db.close()


@router.get("/latest/{run_id}")
async def get_latest_metric(run_id: int):
    if run_id in trainer.active_runs:
        return trainer.active_runs[run_id]

    db = SessionLocal()
    try:
        metric = (
            db.query(TrainingMetric)
            .filter(TrainingMetric.run_id == run_id)
            .order_by(TrainingMetric.epoch.desc())
            .first()
        )
        if not metric:
            raise HTTPException(status_code=404, detail="No metrics found")
        run = db.query(TrainingRun).get(run_id)
        return {
            "epoch": metric.epoch,
            "total_epochs": run.total_epochs if run else metric.epoch,
            "metrics": {
                "loss": metric.loss,
                "q_value_mean": metric.q_value_mean,
                "q_value_max": metric.q_value_max,
                "q_value_min": metric.q_value_min,
                "cql_penalty": metric.cql_penalty,
            },
            "cumulative_reward": metric.cumulative_reward,
        }
    finally:
        db.close()
