"""The browser screens: they serve pages, carry no data, and their scripts are valid (W5.6)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app as agent_app
from evals import api

EVAL_PAGES = ["/ui/experiments", "/ui/experiments/exp_abc", "/ui/runs/erun_abc", "/ui/compare"]
SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


@pytest.fixture
def eval_client() -> TestClient:
    return TestClient(api.app)


@pytest.mark.parametrize("path", EVAL_PAGES)
def test_each_evaluation_screen_is_served_without_a_token(eval_client, path) -> None:
    response = eval_client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="token-form"' in response.text  # the user pastes an operator token


def test_pages_carry_no_data(eval_client) -> None:
    for path in EVAL_PAGES:
        html = eval_client.get(path).text
        assert "refund_eligible_001" not in html and "249900" not in html


def test_the_operator_token_goes_to_the_api_not_into_the_page(eval_client) -> None:
    html = eval_client.get("/ui/experiments").text
    assert "Authorization" in html and "sessionStorage" in html
    assert "/eval/experiments" in html  # the page reads through the API


def test_the_ids_in_the_url_are_data_not_markup(eval_client) -> None:
    hostile = "x<img src=1 onerror=alert(1)>\"';"
    for path in (f"/ui/runs/{hostile}", f"/ui/experiments/{hostile}"):
        html = eval_client.get(path).text
        assert "<img src=1" not in html and 'onerror=alert(1)>"' not in html
        assert "\\" + "u003cimg" in html  # tojson escapes the markup characters


def test_the_run_screen_keeps_the_ground_truth_apart_from_what_the_agent_saw(eval_client) -> None:
    html = eval_client.get("/ui/runs/erun_abc").text
    assert 'class="panel agent-saw"' in html and 'class="panel hidden-truth"' in html
    assert "hidden from the agent" in html  # the label on the ground-truth panel
    assert "private reasoning is not recorded" in html


def test_the_experiment_screen_has_the_failure_shortcuts(eval_client) -> None:
    html = eval_client.get("/ui/experiments/exp_abc").text
    assert "refund failures" in html and "failed runs" in html
    assert "Failure reasons" in html and "Latency and tokens" in html


def test_the_comparison_screen_names_regressed_tasks(eval_client) -> None:
    html = eval_client.get("/ui/compare").text
    assert "Regressed tasks" in html and "Improved tasks" in html


def test_the_ui_routes_are_not_part_of_the_api_schema() -> None:
    assert all(path.startswith("/eval/") for path in api.app.openapi()["paths"])


def test_the_agent_demo_screen_has_its_five_panels() -> None:
    response = TestClient(agent_app).get("/demo")
    assert response.status_code == 200
    for heading in (
        "Customer complaint",
        "Workflow timeline",
        "Evidence and policy",
        "Approval queue",
        "Final outcome",
    ):
        assert heading in response.text
    assert "private reasoning" in response.text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    ("client_app", "path"),
    [(api.app, p) for p in EVAL_PAGES] + [(agent_app, "/demo"), (agent_app, "/review")],
)
def test_every_script_is_valid_javascript(client_app, path, tmp_path: Path) -> None:
    html = TestClient(client_app).get(path).text
    scripts = SCRIPT.findall(html)
    assert scripts
    for index, source in enumerate(scripts):
        file = tmp_path / f"script{index}.js"
        file.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(file)], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, f"{path} script {index}: {result.stderr}"
