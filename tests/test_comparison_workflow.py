from pathlib import Path

from typer.testing import CliRunner

from cognivore.agent.runs import RunStore
from cognivore.cli import app


def test_comparison_can_be_approved_after_reopening(tmp_path: Path) -> None:
    before, after = tmp_path / "before.md", tmp_path / "after.md"
    before.write_text("Timeout is 5 seconds.\n", encoding="utf-8")
    after.write_text("Timeout is 10 seconds.\n", encoding="utf-8")
    data = tmp_path / "data"
    runner = CliRunner(env={"COGNIVORE_DATA_DIR": str(data)})
    staged = runner.invoke(app, ["compare", str(before), str(after)])
    assert staged.exit_code == 0, staged.output
    run_id = staged.output.strip().splitlines()[-1].split()[-1]
    assert not (data / "reports").exists()

    reopened = RunStore(data / "runs.sqlite3").get(run_id)
    assert reopened.status == "pending"
    assert "-Timeout is 5 seconds." in reopened.report
    assert "+Timeout is 10 seconds." in reopened.report

    approved = runner.invoke(app, ["approve-comparison", run_id])
    assert approved.exit_code == 0, approved.output
    output = data / "reports" / f"{run_id}.md"
    assert output.read_text(encoding="utf-8") == reopened.report
    assert RunStore(data / "runs.sqlite3").get(run_id).status == "approved"
    assert runner.invoke(app, ["approve-comparison", run_id]).exit_code == 0
