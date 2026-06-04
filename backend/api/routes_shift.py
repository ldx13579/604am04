import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from backend.database import SessionLocal
from backend.models import ShiftDetectionRecord, ModelSnapshot
from backend.shift_detection import shift_detector
from backend.config import N_ITEMS, N_CATEGORIES

router = APIRouter(prefix="/api/shift", tags=["shift_detection"])


class ShiftDetectionResponse(BaseModel):
    shift_type: str
    metric_name: str
    metric_value: float
    threshold: float
    is_alert: bool
    details: Optional[dict] = None

    class Config:
        from_attributes = True


class ShiftRecordResponse(BaseModel):
    id: int
    detection_time: Optional[str]
    shift_type: str
    metric_name: str
    metric_value: float
    threshold: float
    is_alert: bool
    details: Optional[dict] = None
    triggered_retrain: bool
    retrain_run_id: Optional[int] = None

    class Config:
        from_attributes = True


class QValueDistributionResponse(BaseModel):
    action_indices: List[int]
    q_values: List[float]
    in_distribution_mask: List[bool]


class ModelSnapshotResponse(BaseModel):
    id: int
    run_id: int
    epoch: int
    algorithm: str
    hyperparameters: dict
    state_dim: int
    action_dim: int
    performance_reward: Optional[float]
    created_at: Optional[str]

    class Config:
        from_attributes = True


@router.post("/detect", response_model=List[ShiftDetectionResponse])
async def run_shift_detection(background_tasks: BackgroundTasks):
    """Run distribution shift detection between offline data and online simulation."""
    online_actions, online_rewards, online_states = shift_detector.simulate_online_data(n_episodes=50)
    current_items = set(range(N_ITEMS))

    results = shift_detector.run_detection_and_record(
        online_actions, online_rewards, online_states, current_items
    )
    return [ShiftDetectionResponse(**r) for r in results]


@router.post("/detect_with_new_items", response_model=List[ShiftDetectionResponse])
async def detect_with_new_items(new_item_count: int = 20):
    """Simulate adding new items and check for distribution shift."""
    online_actions, online_rewards, online_states = shift_detector.simulate_online_data(n_episodes=50)
    current_items = set(range(N_ITEMS + new_item_count))

    results = shift_detector.run_detection_and_record(
        online_actions, online_rewards, online_states, current_items
    )
    return [ShiftDetectionResponse(**r) for r in results]


@router.get("/records", response_model=List[ShiftRecordResponse])
async def get_detection_records(limit: int = 50, alerts_only: bool = False):
    """Get historical shift detection records."""
    db = SessionLocal()
    try:
        query = db.query(ShiftDetectionRecord).order_by(ShiftDetectionRecord.id.desc())
        if alerts_only:
            query = query.filter(ShiftDetectionRecord.is_alert == True)
        records = query.limit(limit).all()
        return [
            ShiftRecordResponse(
                id=r.id,
                detection_time=r.detection_time.isoformat() if r.detection_time else None,
                shift_type=r.shift_type,
                metric_name=r.metric_name,
                metric_value=r.metric_value,
                threshold=r.threshold,
                is_alert=r.is_alert,
                details=r.details,
                triggered_retrain=r.triggered_retrain,
                retrain_run_id=r.retrain_run_id,
            )
            for r in records
        ]
    finally:
        db.close()


@router.get("/alerts", response_model=List[ShiftRecordResponse])
async def get_alerts():
    """Get active shift detection alerts."""
    db = SessionLocal()
    try:
        records = db.query(ShiftDetectionRecord).filter(
            ShiftDetectionRecord.is_alert == True
        ).order_by(ShiftDetectionRecord.id.desc()).limit(20).all()
        return [
            ShiftRecordResponse(
                id=r.id,
                detection_time=r.detection_time.isoformat() if r.detection_time else None,
                shift_type=r.shift_type,
                metric_name=r.metric_name,
                metric_value=r.metric_value,
                threshold=r.threshold,
                is_alert=r.is_alert,
                details=r.details,
                triggered_retrain=r.triggered_retrain,
                retrain_run_id=r.retrain_run_id,
            )
            for r in records
        ]
    finally:
        db.close()


@router.get("/q_distribution/{run_id}", response_model=QValueDistributionResponse)
async def get_q_value_distribution(run_id: int):
    """Get Q-value distribution for all actions from a trained model.

    Returns Q-values for each action along with in-distribution/OOD classification.
    """
    db = SessionLocal()
    try:
        snapshot = db.query(ModelSnapshot).filter(
            ModelSnapshot.run_id == run_id
        ).order_by(ModelSnapshot.epoch.desc()).first()

        if not snapshot:
            raise HTTPException(status_code=404, detail="No model snapshot found for this run")

        import torch
        import io
        from backend.algorithms.cql import CQL
        from backend.algorithms.dqn import DQN
        from backend.environment.user_model import UserModel

        buffer = io.BytesIO(snapshot.parameters_blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        if snapshot.algorithm in ("cql", "dqn"):
            agent = CQL(state_dim=snapshot.state_dim, action_dim=snapshot.action_dim)
            agent.q_network.load_state_dict(state_dict["q_network"])
        else:
            raise HTTPException(status_code=400, detail=f"Q-distribution not supported for {snapshot.algorithm}")

        state = UserModel.initial_state()
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.q_network(state_t).squeeze(0).numpy()

        from sqlalchemy import text as sql_text
        result = db.execute(sql_text(
            "SELECT DISTINCT action FROM offline_transitions LIMIT 10000"
        )).fetchall()
        in_dist_actions = set(r[0] for r in result)

        in_distribution_mask = [i in in_dist_actions for i in range(snapshot.action_dim)]

        return QValueDistributionResponse(
            action_indices=list(range(snapshot.action_dim)),
            q_values=q_values.tolist(),
            in_distribution_mask=in_distribution_mask,
        )
    finally:
        db.close()


@router.get("/snapshots/{run_id}", response_model=List[ModelSnapshotResponse])
async def get_model_snapshots(run_id: int):
    """Get all model parameter snapshots for a training run."""
    db = SessionLocal()
    try:
        snapshots = db.query(ModelSnapshot).filter(
            ModelSnapshot.run_id == run_id
        ).order_by(ModelSnapshot.epoch).all()
        return [
            ModelSnapshotResponse(
                id=s.id,
                run_id=s.run_id,
                epoch=s.epoch,
                algorithm=s.algorithm,
                hyperparameters=s.hyperparameters,
                state_dim=s.state_dim,
                action_dim=s.action_dim,
                performance_reward=s.performance_reward,
                created_at=s.created_at.isoformat() if s.created_at else None,
            )
            for s in snapshots
        ]
    finally:
        db.close()
