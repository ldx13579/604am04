from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks
from backend.database import SessionLocal
from backend.models import PerfReportRun
from backend.services.benchmark_runner import benchmark_runner
from backend.schemas import BenchmarkStartRequest, PerfReportEntry, PerfReportResponse

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.post("/benchmark/start")
async def start_benchmark(request: BenchmarkStartRequest, background_tasks: BackgroundTasks):
    if benchmark_runner.is_running:
        return {"status": "already_running"}
    background_tasks.add_task(
        benchmark_runner.run_benchmark,
        request.dataset_sizes,
        request.algorithm,
        request.epochs,
        request.algorithms,
    )
    return {
        "status": "started",
        "dataset_sizes": request.dataset_sizes,
        "algorithms": request.algorithms or [request.algorithm],
    }


@router.get("/benchmark/results", response_model=PerfReportResponse)
async def get_results():
    db = SessionLocal()
    try:
        runs = db.query(PerfReportRun).order_by(
            PerfReportRun.algorithm, PerfReportRun.dataset_size
        ).all()
        entries = [PerfReportEntry.model_validate(r) for r in runs]
        return PerfReportResponse(entries=entries)
    finally:
        db.close()


@router.get("/benchmark/status")
async def get_status():
    return benchmark_runner.get_status()


@router.get("/benchmark/comparison")
async def get_comparison():
    """Cross-algorithm comparison with efficiency metrics."""
    return benchmark_runner.get_comparison()


@router.get("/benchmark/analysis")
async def get_analysis():
    """Comprehensive analysis: scaling behavior, diminishing returns, recommendations."""
    db = SessionLocal()
    try:
        runs = db.query(PerfReportRun).order_by(
            PerfReportRun.algorithm, PerfReportRun.dataset_size
        ).all()
    finally:
        db.close()

    if not runs:
        return {"analysis": None}

    algorithms = {}
    for r in runs:
        if r.algorithm not in algorithms:
            algorithms[r.algorithm] = []
        algorithms[r.algorithm].append({
            "dataset_size": r.dataset_size,
            "time": r.training_time_seconds,
            "reward": r.final_reward,
            "convergence": r.convergence_epoch,
        })

    analysis = {}
    for alg, data in algorithms.items():
        data.sort(key=lambda x: x["dataset_size"])
        sizes = [d["dataset_size"] for d in data]
        rewards = [d["reward"] for d in data]
        times = [d["time"] for d in data]

        marginal_returns = []
        for i in range(1, len(data)):
            reward_gain = rewards[i] - rewards[i - 1]
            size_gain = sizes[i] - sizes[i - 1]
            time_gain = times[i] - times[i - 1]
            marginal_returns.append({
                "from_size": sizes[i - 1],
                "to_size": sizes[i],
                "reward_gain": reward_gain,
                "time_cost": time_gain,
                "efficiency": reward_gain / time_gain if time_gain > 0 else 0,
            })

        best_efficiency_idx = 0
        if marginal_returns:
            best_efficiency_idx = max(
                range(len(marginal_returns)),
                key=lambda i: marginal_returns[i]["efficiency"]
            )

        analysis[alg] = {
            "data_points": data,
            "marginal_returns": marginal_returns,
            "recommended_size": sizes[best_efficiency_idx + 1] if marginal_returns else sizes[0],
            "max_reward": max(rewards),
            "total_time": sum(times),
            "scaling_factor": times[-1] / times[0] if len(times) > 1 and times[0] > 0 else 1.0,
        }

    best_alg = max(analysis.items(), key=lambda x: x[1]["max_reward"])[0] if analysis else None

    return {
        "analysis": analysis,
        "best_algorithm": best_alg,
        "recommendation": f"Use {best_alg} with dataset size {analysis[best_alg]['recommended_size']}" if best_alg else None,
    }
