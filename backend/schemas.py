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


class FQEStartRequest(BaseModel):
    source_run_id: int
    hyperparameters: Optional[dict] = None


class FQEEvaluationResponse(BaseModel):
    id: int
    source_run_id: int
    algorithm: str
    hyperparameters: dict
    status: str
    estimated_value: Optional[float]
    total_epochs: int
    current_epoch: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class FQEMetricResponse(BaseModel):
    epoch: int
    estimated_value: float
    fqe_loss: float

    class Config:
        from_attributes = True


class EnsembleMetricResponse(BaseModel):
    epoch: int
    uncertainty_mean: float
    uncertainty_max: float
    exploration_ratio: float
    per_model_losses: Optional[List[float]] = None
    per_model_q_means: Optional[List[float]] = None

    class Config:
        from_attributes = True


class AlphaSweepRequest(BaseModel):
    alpha_values: List[float]
    base_hyperparameters: Optional[dict] = None


class AlphaComparisonItem(BaseModel):
    run_id: int
    alpha: float
    convergence_epoch: Optional[int]
    final_reward: Optional[float]
    reward_stability: Optional[float]
    mean_cql_penalty: Optional[float]


class AlphaComparisonResponse(BaseModel):
    comparisons: List[AlphaComparisonItem]


class PolicyVersionCreate(BaseModel):
    run_id: int
    snapshot_id: int
    version_tag: str
    stage: str = "candidate"
    fqe_evaluation_id: Optional[int] = None
    notes: Optional[str] = None


class PolicyVersionResponse(BaseModel):
    id: int
    run_id: int
    snapshot_id: int
    version_tag: str
    stage: str
    fqe_evaluation_id: Optional[int]
    notes: Optional[str]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class PolicyVersionStageUpdate(BaseModel):
    stage: str
