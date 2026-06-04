from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TrainingStartRequest(BaseModel):
    algorithm: str
    hyperparameters: Optional[dict] = None


class TrainingRunResponse(BaseModel):
    id: int
    algorithm: str
    hyperparameters: dict
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_epochs: int
    current_epoch: Optional[int] = 0
    best_reward: Optional[float]
    error_detail: Optional[str] = None

    class Config:
        from_attributes = True


class MetricResponse(BaseModel):
    epoch: int
    loss: float
    q_value_mean: Optional[float]
    q_value_max: Optional[float]
    q_value_min: Optional[float]
    cumulative_reward: float
    cql_penalty: Optional[float]

    class Config:
        from_attributes = True


class MetricsListResponse(BaseModel):
    run_id: int
    algorithm: str
    metrics: List[MetricResponse]


class DataStatusResponse(BaseModel):
    is_running: bool
    progress: float
    total_generated: int


class DataStatsResponse(BaseModel):
    total_transitions: int
    total_episodes: int
    avg_reward: float
    reward_std: float
