"""Evaluation task format: validation, the two views, and immutable versions (W4.1)."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from evals import loader
from evals.loader import TaskVersionError, check_version, content_hash, split_views
from evals.task_format import AgentInput, Task

SAMPLE = Path(loader.DATASETS / "support-v1/tasks/refund_eligible_001.yaml")
HIDDEN_KEYS = {"environment_fixture", "reviewer_behavior", "evaluation"}


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))


def clarification(raw: dict) -> dict:
    """The sample turned into a missing-information task: no order id given."""
    raw = copy.deepcopy(raw)
    raw.update(id="missing_order_001", category="missing_information", reviewer_behavior=None)
    raw["agent_input"] = {
        "customer_subject": "auth|alice",
        "user_message": "I still haven't got my refund.",
    }
    raw["evaluation"].update(
        expected_order_id=None,
        resolution="clarify",
        approval_required=False,
        allowed_actions=["list_orders", "update_ticket"],
        forbidden_actions=["create_refund_proposal", "execute_approved_refund"],
        required_behavior=[],
        outcome={"ticket_status": "awaiting_customer", "run_status": "waiting_for_customer"},
    )
    return raw


def all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in all_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in all_keys(v)}
    return set()


def test_sample_task_is_valid_and_locked():
    tasks = loader.load_dataset("support-v1")
    assert "refund_eligible_001" in {t.id for t in tasks}


def test_agent_view_holds_only_agent_input(raw):
    views = split_views(Task.model_validate(raw))
    dumped = views.agent.model_dump()
    assert set(dumped) == {"task_id", "agent_input"}
    assert set(dumped["agent_input"]) == set(AgentInput.model_fields)
    assert not all_keys(dumped) & (HIDDEN_KEYS | {"expected_customer_id", "outcome", "faults"})
    # Nothing hidden leaks through a value either, e.g. the amount the reviewer would approve.
    assert "249900" not in json.dumps(dumped)


def test_evaluator_view_holds_everything_else(raw):
    evaluator = split_views(Task.model_validate(raw)).evaluator
    assert HIDDEN_KEYS <= set(evaluator.model_dump())
    assert "agent_input" not in evaluator.model_dump()
    assert evaluator.content_hash.startswith("sha256:")


@pytest.mark.parametrize(
    "path",
    [(), ("agent_input",), ("environment_fixture",), ("evaluation",), ("evaluation", "outcome")],
)
def test_unknown_keys_are_rejected(raw, path):
    target = raw
    for key in path:
        target = target[key]
    target["unexpected"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Task.model_validate(raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["evaluation"].update(expected_customer_id="cus_bob"), "authenticated"),
        (lambda r: r["agent_input"].update(customer_subject="auth|nobody"), "fixture customer"),
        (lambda r: r.update(reviewer_behavior=None), "reviewer_behavior"),
        (lambda r: r["evaluation"].update(allowed_actions=["issue_cash"]), "unknown steps"),
        (lambda r: r["evaluation"].update(applicable_policy=["refund_policy"]), "section"),
        (lambda r: r["environment_fixture"]["returns"][0].update(order_id="ord_9"), "unknown"),
        (lambda r: r["evaluation"]["outcome"].update(approval_requested=False), "disagree"),
    ],
)
def test_inconsistent_tasks_are_rejected(raw, change, message):
    change(raw)
    with pytest.raises(ValidationError, match=message):
        Task.model_validate(raw)


def test_evaluation_covers_the_section_15_expectations():
    fields = set(Task.model_fields["evaluation"].annotation.model_fields)
    assert {
        "expected_customer_id",
        "expected_order_id",
        "applicable_policy",
        "allowed_actions",
        "forbidden_actions",
        "approval_required",
        "outcome",
        "resolution",  # escalation or clarification
    } <= fields


def test_clarification_task_stops_for_the_customer(raw):
    task = Task.model_validate(clarification(raw))
    assert task.evaluation.outcome.ticket_status == "awaiting_customer"


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["evaluation"]["outcome"].update(
            ticket_status="resolved", run_status="completed"
        ),
        lambda r: r["evaluation"]["outcome"].update(refund_count=1, refund_total_minor=100),
        lambda r: r["evaluation"].update(resolution="resolve"),
    ],
)
def test_clarification_task_must_be_side_effect_free(raw, change):
    raw = clarification(raw)
    change(raw)
    with pytest.raises(ValidationError, match="clarification"):
        Task.model_validate(raw)


def test_hash_ignores_formatting_but_not_content(raw):
    task = Task.model_validate(raw)
    reordered = Task.model_validate(dict(reversed(list(raw.items()))))
    assert content_hash(reordered) == content_hash(task)
    raw["agent_input"]["user_message"] += " Please hurry."
    assert content_hash(Task.model_validate(raw)) != content_hash(task)


def test_locked_version_cannot_be_edited_in_place(raw):
    task = Task.model_validate(raw)
    lock = {task.id: {"1": content_hash(task)}}
    assert check_version(task, lock)

    raw["evaluation"]["outcome"]["refund_total_minor"] = 100000
    edited = Task.model_validate(raw)
    with pytest.raises(TaskVersionError, match="bump its version"):
        check_version(edited, lock)

    raw["version"] = 2
    assert not check_version(Task.model_validate(raw), lock)  # a new version, not yet locked


def test_version_cannot_go_backwards(raw):
    task = Task.model_validate(raw)
    with pytest.raises(TaskVersionError, match="older"):
        check_version(task, {task.id: {"2": "sha256:x"}})


def test_unlocked_task_fails_loading(tmp_path, monkeypatch, raw):
    (tmp_path / "ds/tasks").mkdir(parents=True)
    (tmp_path / "ds/tasks/refund_eligible_001.yaml").write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(loader, "DATASETS", tmp_path)
    with pytest.raises(TaskVersionError, match="not locked"):
        loader.load_dataset("ds")
    assert loader.lock_dataset("ds") == ["refund_eligible_001 v1"]
    assert loader.lock_dataset("ds") == []
    assert len(loader.load_dataset("ds")) == 1


def test_file_name_must_match_task_id(tmp_path, raw):
    path = tmp_path / "other_name.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="file name"):
        loader.parse_task(path)


def test_app_never_imports_evals():
    app = Path(loader.__file__).resolve().parents[1] / "app"
    offenders = [
        p
        for p in app.rglob("*.py")
        if "import evals" in (text := p.read_text(encoding="utf-8")) or "from evals" in text
    ]
    assert offenders == []
