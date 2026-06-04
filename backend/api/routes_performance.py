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
    )
    return {"status": "started", "dataset_sizes": request.dataset_sizes}


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
