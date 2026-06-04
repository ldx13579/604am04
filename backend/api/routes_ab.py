from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from scipy import stats
from backend.database import SessionLocal
from backend.models import ABExperiment, ABImpression, ABClick, ABCTRSnapshot
from backend.schemas import (
    ABExperimentCreateRequest, ABExperimentResponse,
    ABCTRDataPoint, ABSummaryResponse
)

router = APIRouter(prefix="/api/ab", tags=["ab_testing"])


@router.post("/experiments", response_model=ABExperimentResponse)
async def create_experiment(request: ABExperimentCreateRequest):
    db = SessionLocal()
    try:
        experiment = ABExperiment(
            name=request.name,
            strategy_a=request.strategy_a,
            strategy_b=request.strategy_b,
            traffic_split=request.traffic_split,
            status="running",
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        return ABExperimentResponse.model_validate(experiment)
    finally:
        db.close()


@router.get("/experiments")
async def list_experiments():
    db = SessionLocal()
    try:
        experiments = db.query(ABExperiment).order_by(ABExperiment.created_at.desc()).all()
        return [ABExperimentResponse.model_validate(e) for e in experiments]
    finally:
        db.close()


@router.put("/experiments/{experiment_id}/status")
async def update_experiment_status(experiment_id: int, status: str):
    if status not in ("running", "paused", "completed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    db = SessionLocal()
    try:
        experiment = db.query(ABExperiment).get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        experiment.status = status
        if status == "completed":
            experiment.ended_at = datetime.utcnow()
        db.commit()
        return {"status": status}
    finally:
        db.close()


@router.get("/experiments/{experiment_id}/ctr")
async def get_ctr(experiment_id: int, window_minutes: int = 5, limit: int = 100):
    db = SessionLocal()
    try:
        snapshots = db.query(ABCTRSnapshot).filter(
            ABCTRSnapshot.experiment_id == experiment_id
        ).order_by(ABCTRSnapshot.window_start.desc()).limit(limit).all()

        return [ABCTRDataPoint(
            window_start=s.window_start,
            group_name=s.group_name,
            impressions_count=s.impressions_count,
            clicks_count=s.clicks_count,
            ctr=s.ctr,
        ) for s in reversed(snapshots)]
    finally:
        db.close()


@router.get("/experiments/{experiment_id}/summary", response_model=ABSummaryResponse)
async def get_summary(experiment_id: int):
    db = SessionLocal()
    try:
        experiment = db.query(ABExperiment).get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        impressions_a = db.query(ABImpression).filter(
            ABImpression.experiment_id == experiment_id,
            ABImpression.group_name == "A"
        ).count()
        impressions_b = db.query(ABImpression).filter(
            ABImpression.experiment_id == experiment_id,
            ABImpression.group_name == "B"
        ).count()

        clicks_a = db.query(ABClick).join(ABImpression).filter(
            ABImpression.experiment_id == experiment_id,
            ABImpression.group_name == "A",
            ABClick.clicked == True
        ).count()
        clicks_b = db.query(ABClick).join(ABImpression).filter(
            ABImpression.experiment_id == experiment_id,
            ABImpression.group_name == "B",
            ABClick.clicked == True
        ).count()

        ctr_a = clicks_a / impressions_a if impressions_a > 0 else 0.0
        ctr_b = clicks_b / impressions_b if impressions_b > 0 else 0.0
        lift = ((ctr_a - ctr_b) / ctr_b * 100) if ctr_b > 0 else 0.0

        p_value = None
        is_significant = False
        if impressions_a >= 30 and impressions_b >= 30:
            p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)
            if p_pool > 0 and p_pool < 1:
                se = (p_pool * (1 - p_pool) * (1/impressions_a + 1/impressions_b)) ** 0.5
                z = (ctr_a - ctr_b) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                is_significant = p_value < 0.05

        return ABSummaryResponse(
            experiment_id=experiment_id,
            group_a_impressions=impressions_a,
            group_a_clicks=clicks_a,
            group_a_ctr=ctr_a,
            group_b_impressions=impressions_b,
            group_b_clicks=clicks_b,
            group_b_ctr=ctr_b,
            lift_percent=lift,
            p_value=p_value,
            is_significant=is_significant,
        )
    finally:
        db.close()
