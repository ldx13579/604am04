import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes_data import router as data_router
from backend.api.routes_training import router as training_router
from backend.api.routes_metrics import router as metrics_router
from backend.api.routes_shift import router as shift_router
from backend.api.routes_evaluation import router as evaluation_router
from backend.api.routes_recommend import router as recommend_router
from backend.api.routes_ab import router as ab_router
from backend.api.routes_finetune import router as finetune_router
from backend.api.routes_performance import router as performance_router
from backend.algorithms.trainer import trainer
from backend.database import engine, Base

app = FastAPI(title="Offline RL Recommendation Simulator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(training_router)
app.include_router(metrics_router)
app.include_router(shift_router)
app.include_router(evaluation_router)
app.include_router(recommend_router)
app.include_router(ab_router)
app.include_router(finetune_router)
app.include_router(performance_router)


@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
    from backend.data.generator import data_generator
    data_generator.restore_state()

    from apscheduler.schedulers.background import BackgroundScheduler
    from backend.services.ab_aggregator import ab_aggregator
    from backend.services.online_finetuner import online_finetuner
    from backend.config import FINETUNE_CONFIG, AB_TEST_CONFIG

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        ab_aggregator.aggregate,
        'interval',
        seconds=AB_TEST_CONFIG["aggregation_interval_seconds"],
        id='ab_ctr_aggregation',
    )
    scheduler.add_job(
        online_finetuner.run_finetune,
        'interval',
        seconds=FINETUNE_CONFIG["interval_seconds"],
        id='online_finetune',
    )
    scheduler.start()
    app.state.scheduler = scheduler


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: int):
    await websocket.accept()
    try:
        while True:
            if run_id in trainer.active_runs:
                data = trainer.active_runs[run_id]
                await websocket.send_json(data)
            else:
                await websocket.send_json({"status": "idle", "run_id": run_id})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
