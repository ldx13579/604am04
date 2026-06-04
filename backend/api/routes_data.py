import asyncio
from fastapi import APIRouter, BackgroundTasks
from backend.data.generator import data_generator
from backend.schemas import DataStatusResponse, DataStatsResponse
from backend.database import SessionLocal
from sqlalchemy import text

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/generate")
async def generate_data(background_tasks: BackgroundTasks):
    if data_generator.is_running:
        return {"message": "Data generation already in progress", "progress": data_generator.progress}

    background_tasks.add_task(data_generator.generate)
    return {"message": "Data generation started", "total": 1_000_000}


@router.get("/status", response_model=DataStatusResponse)
async def get_data_status():
    return DataStatusResponse(
        is_running=data_generator.is_running,
        progress=data_generator.progress,
        total_generated=data_generator.total_generated,
    )


@router.get("/stats", response_model=DataStatsResponse)
async def get_data_stats():
    db = SessionLocal()
    try:
        result = db.execute(text(
            "SELECT COUNT(*), COUNT(DISTINCT episode_id), AVG(reward), STDDEV(reward) "
            "FROM offline_transitions"
        )).fetchone()
        return DataStatsResponse(
            total_transitions=result[0] or 0,
            total_episodes=result[1] or 0,
            avg_reward=float(result[2] or 0),
            reward_std=float(result[3] or 0),
        )
    finally:
        db.close()
