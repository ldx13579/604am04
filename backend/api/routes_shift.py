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


class QValueHistogramResponse(BaseModel):
    in_dist_values: List[float]
    ood_values: List[float]
    in_dist_mean: float
    ood_mean: float
    in_dist_std: float
    ood_std: float
    bin_edges: List[float]
    in_dist_counts: List[int]
    ood_counts: List[int]


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
    Supports CQL, DQN, and CQL_RNN models.
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
        from backend.algorithms.cql_rnn import CQL_RNN
        from backend.environment.user_model import UserModel

        buffer = io.BytesIO(snapshot.parameters_blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        state = UserModel.initial_state()

        if snapshot.algorithm == "cql_rnn":
            agent = CQL_RNN(state_dim=snapshot.state_dim, action_dim=snapshot.action_dim)
            agent.load_state_dict(state_dict)
            q_values = agent.get_q_distribution(state)
        elif snapshot.algorithm in ("cql", "dqn"):
            agent = CQL(state_dim=snapshot.state_dim, action_dim=snapshot.action_dim)
            agent.q_network.load_state_dict(state_dict["q_network"])
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_t).squeeze(0).numpy()
        else:
            raise HTTPException(status_code=400, detail=f"Q-distribution not supported for {snapshot.algorithm}")

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


@router.get("/q_histogram/{run_id}", response_model=QValueHistogramResponse)
async def get_q_value_histogram(run_id: int, n_states: int = 50, n_bins: int = 30):
    """Get binned Q-value histogram comparing in-distribution vs OOD actions.

    Samples multiple states and aggregates Q-values to produce a histogram
    that clearly shows the conservatism gap between in-dist and OOD actions.
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
        from backend.algorithms.cql_rnn import CQL_RNN
        from backend.environment.user_model import UserModel

        buffer = io.BytesIO(snapshot.parameters_blob)
        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)

        all_q_values = []
        if snapshot.algorithm == "cql_rnn":
            agent = CQL_RNN(state_dim=snapshot.state_dim, action_dim=snapshot.action_dim)
            agent.load_state_dict(state_dict)
            for _ in range(n_states):
                state = UserModel.initial_state()
                q_vals = agent.get_q_distribution(state)
                all_q_values.append(q_vals)
        elif snapshot.algorithm in ("cql", "dqn"):
            agent = CQL(state_dim=snapshot.state_dim, action_dim=snapshot.action_dim)
            agent.q_network.load_state_dict(state_dict["q_network"])
            states = np.array([UserModel.initial_state() for _ in range(n_states)])
            states_t = torch.tensor(states, dtype=torch.float32)
            with torch.no_grad():
                q_batch = agent.q_network(states_t).numpy()
            all_q_values = list(q_batch)
        else:
            raise HTTPException(status_code=400, detail=f"Q-histogram not supported for {snapshot.algorithm}")

        from sqlalchemy import text as sql_text
        result = db.execute(sql_text(
            "SELECT DISTINCT action FROM offline_transitions LIMIT 10000"
        )).fetchall()
        in_dist_actions = set(r[0] for r in result)

        q_matrix = np.array(all_q_values)
        in_dist_mask = np.array([i in in_dist_actions for i in range(snapshot.action_dim)])
        ood_mask = ~in_dist_mask

        in_dist_values = q_matrix[:, in_dist_mask].flatten()
        ood_values = q_matrix[:, ood_mask].flatten()

        all_values = np.concatenate([in_dist_values, ood_values])
        bin_edges = np.linspace(all_values.min(), all_values.max(), n_bins + 1)

        in_dist_counts, _ = np.histogram(in_dist_values, bins=bin_edges)
        ood_counts, _ = np.histogram(ood_values, bins=bin_edges)

        return QValueHistogramResponse(
            in_dist_values=in_dist_values.tolist()[:500],
            ood_values=ood_values.tolist()[:500],
            in_dist_mean=float(in_dist_values.mean()),
            ood_mean=float(ood_values.mean()),
            in_dist_std=float(in_dist_values.std()),
            ood_std=float(ood_values.std()),
            bin_edges=bin_edges.tolist(),
            in_dist_counts=in_dist_counts.tolist(),
            ood_counts=ood_counts.tolist(),
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
