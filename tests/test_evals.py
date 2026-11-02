import csv
from pathlib import Path

from home_service_agent.evals import run_evals


def test_offline_eval_run_generates_passing_table_and_timeout_trace(tmp_path: Path) -> None:
    results = run_evals(tmp_path)

    assert len(results) >= 15
    assert all(result.passed for result in results)
    table = tmp_path / "eval_table.csv"
    assert table.exists()
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(results)
    assert {row["scenario"] for row in rows} >= {"prompt-injection", "timeout-after-success", "limit-exhaustion"}

    trace = tmp_path / "traces" / "timeout-after-success.jsonl"
    assert trace.exists()
    content = trace.read_text(encoding="utf-8")
    assert "tool_call" in content

