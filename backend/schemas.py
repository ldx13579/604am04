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


# --- Recommendation Service ---

class RecommendRequest(BaseModel):
    user_state: List[float]
    top_k: int = 5
    session_id: Optional[str] = None


class RecommendResponse(BaseModel):
    items: List[int]
    scores: List[float]
    policy_version_id: Optional[int] = None
    group: Optional[str] = None
    impression_id: Optional[int] = None


class FeedbackRequest(BaseModel):
    impression_id: int
    item_id: int
    clicked: bool


class PolicyInfoResponse(BaseModel):
    is_loaded: bool
    policy_version_id: Optional[int] = None
    algorithm: Optional[str] = None


# --- A/B Testing ---

class ABExperimentCreateRequest(BaseModel):
    name: str
    strategy_a: str = "cql"
    strategy_b: str = "random"
    traffic_split: float = 0.5


class ABExperimentResponse(BaseModel):
    id: int
    name: str
    strategy_a: str
    strategy_b: str
    traffic_split: float
    status: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ABCTRDataPoint(BaseModel):
    window_start: datetime
    group_name: str
    impressions_count: int
    clicks_count: int
    ctr: float


class ABSummaryResponse(BaseModel):
    experiment_id: int
    group_a_impressions: int
    group_a_clicks: int
    group_a_ctr: float
    group_b_impressions: int
    group_b_clicks: int
    group_b_ctr: float
    lift_percent: float
    p_value: Optional[float] = None
    is_significant: bool


# --- Online Fine-tuning ---

class FinetuneStatusResponse(BaseModel):
    is_running: bool
    buffer_size: int
    last_run_at: Optional[datetime] = None
    last_reward_improvement: Optional[float] = None


class FinetuneRunResponse(BaseModel):
    id: int
    status: str
    n_interactions_used: int
    reward_before: Optional[float] = None
    reward_after: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class FinetuneConfigUpdateRequest(BaseModel):
    interval_seconds: Optional[int] = None
    min_buffer_size: Optional[int] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None


# --- Performance Benchmark ---

class BenchmarkStartRequest(BaseModel):
    dataset_sizes: List[int] = [10000, 50000, 100000, 500000, 1000000]
    algorithm: str = "cql"
    epochs: int = 50


class PerfReportEntry(BaseModel):
    dataset_size: int
    algorithm: str
    training_time_seconds: float
    convergence_epoch: Optional[int] = None
    final_reward: float
    total_epochs_run: int

    class Config:
        from_attributes = True


class PerfReportResponse(BaseModel):
    entries: List[PerfReportEntry]
