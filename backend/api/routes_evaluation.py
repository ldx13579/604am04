from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import List
import numpy as np
from backend.database import SessionLocal
from backend.models import (
    TrainingRun, TrainingMetric, EnsembleMetric, FQEEvaluation, FQEMetric,
    ModelSnapshot, PolicyVersion,
)
from backend.schemas import (
    FQEStartRequest, FQEEvaluationResponse, FQEMetricResponse,
    EnsembleMetricResponse, AlphaSweepRequest, AlphaComparisonItem,
    AlphaComparisonResponse, PolicyVersionCreate, PolicyVersionResponse,
    PolicyVersionStageUpdate,
)
from backend.evaluation.fqe_runner import fqe_runner
from backend.algorithms.trainer import trainer

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("/fqe/start", response_model=FQEEvaluationResponse)
async def start_fqe(request: FQEStartRequest, background_tasks: BackgroundTasks):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(request.source_run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Source run not found")
        if run.status != "completed":
            raise HTTPException(status_code=400, detail=f"Source run not completed (status: {run.status})")

        from backend.config import FQE_DEFAULTS
        merged = {**FQE_DEFAULTS, **(request.hyperparameters or {})}

        evaluation = FQEEvaluation(
            source_run_id=request.source_run_id,
            algorithm=run.algorithm,
            hyperparameters=merged,
            status="pending",
            total_epochs=merged.get("epochs", 50),
            current_epoch=0,
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)
        response = FQEEvaluationResponse.model_validate(evaluation)
    finally:
        db.close()

    background_tasks.add_task(
        fqe_runner.start_evaluation, request.source_run_id,
        request.hyperparameters, evaluation.id
    )
    return response


@router.get("/fqe/results/{evaluation_id}")
async def get_fqe_results(evaluation_id: int):
    db = SessionLocal()
    try:
        evaluation = db.query(FQEEvaluation).get(evaluation_id)
        if not evaluation:
            raise HTTPException(status_code=404, detail="FQE evaluation not found")

        metrics = db.query(FQEMetric).filter(
            FQEMetric.evaluation_id == evaluation_id
        ).order_by(FQEMetric.epoch).all()

        return {
            "evaluation": FQEEvaluationResponse.model_validate(evaluation),
            "metrics": [FQEMetricResponse.model_validate(m) for m in metrics],
        }
    finally:
        db.close()


@router.get("/fqe/runs", response_model=List[FQEEvaluationResponse])
async def list_fqe_runs():
    db = SessionLocal()
    try:
        evaluations = db.query(FQEEvaluation).order_by(FQEEvaluation.id.desc()).all()
        return [FQEEvaluationResponse.model_validate(e) for e in evaluations]
    finally:
        db.close()


@router.get("/fqe/latest/{evaluation_id}")
async def get_fqe_latest(evaluation_id: int):
    if evaluation_id in fqe_runner.active_evaluations:
        return fqe_runner.active_evaluations[evaluation_id]

    db = SessionLocal()
    try:
        evaluation = db.query(FQEEvaluation).get(evaluation_id)
        if not evaluation:
            raise HTTPException(status_code=404, detail="FQE evaluation not found")
        return {
            "epoch": evaluation.current_epoch,
            "total_epochs": evaluation.total_epochs,
            "estimated_value": evaluation.estimated_value,
            "status": evaluation.status,
        }
    finally:
        db.close()


@router.get("/ensemble/uncertainty/{run_id}", response_model=List[EnsembleMetricResponse])
async def get_ensemble_uncertainty(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.algorithm != "ensemble_cql":
            raise HTTPException(status_code=400, detail="Run is not an ensemble_cql run")

        metrics = db.query(EnsembleMetric).filter(
            EnsembleMetric.run_id == run_id
        ).order_by(EnsembleMetric.epoch).all()

        return [EnsembleMetricResponse.model_validate(m) for m in metrics]
    finally:
        db.close()


@router.get("/hyperparams/alpha_comparison", response_model=AlphaComparisonResponse)
async def alpha_comparison(run_ids: str = Query(..., description="Comma-separated run IDs")):
    ids = [int(x.strip()) for x in run_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="No run IDs provided")

    db = SessionLocal()
    try:
        comparisons = []
        for run_id in ids:
            run = db.query(TrainingRun).get(run_id)
            if not run:
                continue

            alpha = run.hyperparameters.get("alpha", 1.0) if run.hyperparameters else 1.0

            metrics = db.query(TrainingMetric).filter(
                TrainingMetric.run_id == run_id
            ).order_by(TrainingMetric.epoch).all()

            if not metrics:
                comparisons.append(AlphaComparisonItem(
                    run_id=run_id, alpha=alpha,
                    convergence_epoch=None, final_reward=None,
                    reward_stability=None, mean_cql_penalty=None,
                ))
                continue

            rewards = [m.cumulative_reward for m in metrics]
            best_reward = max(rewards) if rewards else 0.0

            convergence_epoch = None
            threshold = best_reward * 0.9
            for m in metrics:
                if m.cumulative_reward >= threshold:
                    convergence_epoch = m.epoch
                    break

            last_20_rewards = rewards[-20:] if len(rewards) >= 20 else rewards
            reward_stability = float(np.std(last_20_rewards)) if last_20_rewards else None

            last_10_penalties = [
                m.cql_penalty for m in metrics[-10:]
                if m.cql_penalty is not None
            ]
            mean_cql_penalty = float(np.mean(last_10_penalties)) if last_10_penalties else None

            comparisons.append(AlphaComparisonItem(
                run_id=run_id,
                alpha=alpha,
                convergence_epoch=convergence_epoch,
                final_reward=run.best_reward,
                reward_stability=reward_stability,
                mean_cql_penalty=mean_cql_penalty,
            ))

        return AlphaComparisonResponse(comparisons=comparisons)
    finally:
        db.close()


@router.post("/hyperparams/alpha_sweep")
async def start_alpha_sweep(request: AlphaSweepRequest, background_tasks: BackgroundTasks):
    if not request.alpha_values:
        raise HTTPException(status_code=400, detail="No alpha values provided")
    if len(request.alpha_values) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 alpha values allowed")

    from backend.config import DEFAULT_HYPERPARAMS
    base = {**DEFAULT_HYPERPARAMS.get("cql", {}), **(request.base_hyperparameters or {})}

    run_ids = []
    db = SessionLocal()
    try:
        for alpha in request.alpha_values:
            merged = {**base, "alpha": alpha}
            run = TrainingRun(
                algorithm="cql",
                hyperparameters=merged,
                status="pending",
                total_epochs=merged.get("epochs", 200),
                current_epoch=0,
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            run_ids.append(run.id)
    finally:
        db.close()

    for i, run_id in enumerate(run_ids):
        alpha = request.alpha_values[i]
        hyperparams = {**(request.base_hyperparameters or {}), "alpha": alpha}
        background_tasks.add_task(trainer.start_training, "cql", hyperparams, run_id)

    return {"message": f"Started {len(run_ids)} training runs", "run_ids": run_ids}


@router.post("/versions", response_model=PolicyVersionResponse)
async def create_policy_version(request: PolicyVersionCreate):
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).get(request.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Training run not found")

        snapshot = db.query(ModelSnapshot).get(request.snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Model snapshot not found")
        if snapshot.run_id != request.run_id:
            raise HTTPException(status_code=400, detail="Snapshot does not belong to the specified run")

        if request.stage not in ("candidate", "staging", "production", "archived"):
            raise HTTPException(status_code=400, detail="Invalid stage")

        if request.fqe_evaluation_id:
            fqe = db.query(FQEEvaluation).get(request.fqe_evaluation_id)
            if not fqe:
                raise HTTPException(status_code=404, detail="FQE evaluation not found")

        version = PolicyVersion(
            run_id=request.run_id,
            snapshot_id=request.snapshot_id,
            version_tag=request.version_tag,
            stage=request.stage,
            fqe_evaluation_id=request.fqe_evaluation_id,
            notes=request.notes,
        )
        db.add(version)
        db.commit()
        db.refresh(version)
        return PolicyVersionResponse.model_validate(version)
    finally:
        db.close()


@router.get("/versions", response_model=List[PolicyVersionResponse])
async def list_policy_versions():
    db = SessionLocal()
    try:
        versions = db.query(PolicyVersion).order_by(PolicyVersion.id.desc()).all()
        return [PolicyVersionResponse.model_validate(v) for v in versions]
    finally:
        db.close()


@router.put("/versions/{version_id}/stage", response_model=PolicyVersionResponse)
async def update_version_stage(version_id: int, request: PolicyVersionStageUpdate):
    if request.stage not in ("candidate", "staging", "production", "archived"):
        raise HTTPException(status_code=400, detail="Invalid stage")

    db = SessionLocal()
    try:
        version = db.query(PolicyVersion).get(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Policy version not found")

        if request.stage == "production":
            current_prod = db.query(PolicyVersion).filter(
                PolicyVersion.stage == "production"
            ).all()
            for v in current_prod:
                v.stage = "archived"

        version.stage = request.stage
        db.commit()
        db.refresh(version)
        return PolicyVersionResponse.model_validate(version)
    finally:
        db.close()


@router.get("/versions/compare")
async def compare_versions(ids: str = Query(..., description="Comma-separated version IDs")):
    version_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
    if len(version_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 version IDs required")

    db = SessionLocal()
    try:
        results = []
        for vid in version_ids:
            version = db.query(PolicyVersion).get(vid)
            if not version:
                continue

            run = db.query(TrainingRun).get(version.run_id)
            fqe_value = None
            if version.fqe_evaluation_id:
                fqe = db.query(FQEEvaluation).get(version.fqe_evaluation_id)
                if fqe:
                    fqe_value = fqe.estimated_value

            results.append({
                "version_id": version.id,
                "version_tag": version.version_tag,
                "stage": version.stage,
                "algorithm": run.algorithm if run else None,
                "hyperparameters": run.hyperparameters if run else {},
                "best_reward": run.best_reward if run else None,
                "fqe_estimated_value": fqe_value,
                "snapshot_epoch": version.snapshot.epoch if version.snapshot else None,
                "snapshot_reward": version.snapshot.performance_reward if version.snapshot else None,
            })

        return {"comparisons": results}
    finally:
        db.close()
