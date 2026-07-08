import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cloud_agent.agent import schedule_task


async def _run_schedule_and_wait():
    started = []

    async def sample_task():
        started.append("done")

    await schedule_task(sample_task, delay_seconds=0.05)
    await asyncio.sleep(0.2)
    return started


def test_schedule_task_runs_after_delay():
    started = asyncio.run(_run_schedule_and_wait())
    assert started == ["done"]


def test_schedule_delayed_task_accepts_real_callback():
    async def callback():
        return {"status": "ok"}

    result = asyncio.run(schedule_task(callback, delay_seconds=0.01))
    assert result["status"] == "completed"
