from sqlalchemy import Column, Integer, BigInteger, Float, Boolean, String, Text, DateTime, ForeignKey, func, LargeBinary
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from backend.database import Base


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, nullable=False, index=True)
    popularity = Column(Float, nullable=False, default=1.0)
    embedding = Column(ARRAY(Float), nullable=False)


class OfflineTransition(Base):
    __tablename__ = "offline_transitions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    state = Column(ARRAY(Float), nullable=False)
    action = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    next_state = Column(ARRAY(Float), nullable=False)
    done = Column(Boolean, nullable=False)
    episode_id = Column(Integer, nullable=False, index=True)
    timestamp_step = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class GenerationStatus(Base):
    __tablename__ = "generation_status"

    id = Column(Integer, primary_key=True, default=1)
    is_running = Column(Boolean, nullable=False, default=False)
    progress = Column(Float, nullable=False, default=0.0)
    total_generated = Column(Integer, nullable=False, default=0)
    target_count = Column(Integer, nullable=False, default=1000000)
    last_episode_id = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm = Column(String(50), nullable=False, index=True)
    hyperparameters = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    total_epochs = Column(Integer, nullable=False, default=0)
    current_epoch = Column(Integer, nullable=False, default=0)
    best_reward = Column(Float, nullable=True)
    error_detail = Column(Text, nullable=True)

    metrics = relationship("TrainingMetric", back_populates="run", cascade="all, delete-orphan")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    q_value_mean = Column(Float, nullable=True)
    q_value_max = Column(Float, nullable=True)
    q_value_min = Column(Float, nullable=True)
    cumulative_reward = Column(Float, nullable=False)
    cql_penalty = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("TrainingRun", back_populates="metrics")


class UserBehaviorSequence(Base):
    """Stores user temporal behavior sequences for LSTM encoding."""
    __tablename__ = "user_behavior_sequences"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    episode_id = Column(Integer, nullable=False, index=True)
    step = Column(Integer, nullable=False)
    action = Column(Integer, nullable=False)
    clicked = Column(Boolean, nullable=False)
    dwell_time = Column(Float, nullable=False, default=0.0)
    purchased = Column(Boolean, nullable=False, default=False)
    state = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ModelSnapshot(Base):
    """Stores model parameter snapshots for auditing."""
    __tablename__ = "model_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    algorithm = Column(String(50), nullable=False)
    parameters_blob = Column(LargeBinary, nullable=False)
    hyperparameters = Column(JSONB, nullable=False)
    state_dim = Column(Integer, nullable=False)
    action_dim = Column(Integer, nullable=False)
    performance_reward = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("TrainingRun")


class ShiftDetectionRecord(Base):
    """Records data distribution shift detection results."""
    __tablename__ = "shift_detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_time = Column(DateTime, server_default=func.now())
    shift_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_alert = Column(Boolean, nullable=False, default=False)
    details = Column(JSONB, nullable=True)
    triggered_retrain = Column(Boolean, nullable=False, default=False)
    retrain_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    retrain_run = relationship("TrainingRun")


class EnsembleMetric(Base):
    """Per-epoch uncertainty metrics for ensemble CQL training runs."""
    __tablename__ = "ensemble_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    uncertainty_mean = Column(Float, nullable=False)
    uncertainty_max = Column(Float, nullable=False)
    exploration_ratio = Column(Float, nullable=False)
    per_model_losses = Column(JSONB, nullable=True)
    per_model_q_means = Column(JSONB, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("TrainingRun")


class FQEEvaluation(Base):
    """Fitted Q Evaluation run metadata."""
    __tablename__ = "fqe_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    algorithm = Column(String(50), nullable=False)
    hyperparameters = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    estimated_value = Column(Float, nullable=True)
    total_epochs = Column(Integer, nullable=False, default=50)
    current_epoch = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    source_run = relationship("TrainingRun")
    metrics = relationship("FQEMetric", back_populates="evaluation", cascade="all, delete-orphan")


class FQEMetric(Base):
    """Per-epoch FQE convergence metrics."""
    __tablename__ = "fqe_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("fqe_evaluations.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    estimated_value = Column(Float, nullable=False)
    fqe_loss = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    evaluation = relationship("FQEEvaluation", back_populates="metrics")


class PolicyVersion(Base):
    """Versioned policy snapshots for comparison and lifecycle management."""
    __tablename__ = "policy_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False, index=True)
    snapshot_id = Column(Integer, ForeignKey("model_snapshots.id"), nullable=False)
    version_tag = Column(String(50), nullable=False)
    stage = Column(String(20), nullable=False, default="candidate")
    fqe_evaluation_id = Column(Integer, ForeignKey("fqe_evaluations.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    run = relationship("TrainingRun")
    snapshot = relationship("ModelSnapshot")
    fqe_evaluation = relationship("FQEEvaluation")
