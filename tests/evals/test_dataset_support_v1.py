"""The support-v1 dataset: size, category mix, splits and policy citations (W4.2)."""

import re
from collections import Counter
from pathlib import Path

import pytest

from evals import loader

DATASET = "support-v1"
POLICIES = Path(loader.__file__).resolve().parents[1] / "app/policies/documents"
CATEGORY_COUNTS = {
    "read_only_lookup": 7,
    "eligible_refund": 7,
    "ineligible_refund": 7,
    "damaged_item": 5,
    "missing_information": 6,
    "access_boundary": 5,
    "tool_failure": 6,
    "duplicate_operation": 3,
    "untrusted_content": 4,
}
SPLITS = ("dev", "validation", "test")


@pytest.fixture(scope="module")
def tasks() -> dict[str, loader.LoadedTask]:
    return {task.id: task for task in loader.load_dataset(DATASET)}


def test_fifty_tasks_in_the_planned_category_mix(tasks):
    assert len(tasks) == 50
    assert Counter(t.evaluator.category for t in tasks.values()) == CATEGORY_COUNTS


def test_every_task_is_in_exactly_one_split(tasks):
    listed = [task_id for split in SPLITS for task_id in loader.split_ids(DATASET, split)]
    assert sorted(listed) == sorted(tasks)


def test_splits_are_by_scenario_family(tasks):
    split_of_family: dict[str, set[str]] = {}
    for split in SPLITS:
        for task_id in loader.split_ids(DATASET, split):
            family = tasks[task_id].evaluator.scenario_family
            split_of_family.setdefault(family, set()).add(split)
    assert {f: s for f, s in split_of_family.items() if len(s) > 1} == {}


def test_split_sizes_are_about_60_20_20():
    sizes = {split: len(loader.split_ids(DATASET, split)) for split in SPLITS}
    assert sizes["test"] == 10
    assert 8 <= sizes["validation"] <= 12
    assert 28 <= sizes["dev"] <= 32


def test_pilot_is_five_dev_tasks_one_per_core_category(tasks):
    pilot = loader.split_ids(DATASET, "pilot")
    assert set(pilot) <= set(loader.split_ids(DATASET, "dev"))
    assert sorted(tasks[t].evaluator.category for t in pilot) == [
        "access_boundary",
        "eligible_refund",
        "ineligible_refund",
        "read_only_lookup",
        "tool_failure",
    ]
    assert len(loader.load_split(DATASET, "pilot")) == 5


def test_policy_citations_point_at_real_sections(tasks):
    for task in tasks.values():
        for ref in task.evaluator.evaluation.applicable_policy:
            document, section = ref.split(":section_")
            text = (POLICIES / f"{document}.md").read_text(encoding="utf-8")
            assert re.search(rf"^## Section {section}:", text, re.MULTILINE), f"{task.id}: {ref}"


def test_every_task_relies_on_a_cited_policy_unless_it_only_asks(tasks):
    # Clarifying which order is meant is a workflow rule (W2.13), not a policy decision.
    uncited = [
        t.id
        for t in tasks.values()
        if not t.evaluator.evaluation.applicable_policy
        and t.evaluator.category != "missing_information"
    ]
    assert uncited == []


def test_every_task_file_starts_with_its_review_note():
    for path in (loader.DATASETS / DATASET / "tasks").glob("*.yaml"):
        assert path.read_text(encoding="utf-8").startswith("# "), path.name


def test_refund_tasks_expect_exactly_the_permitted_amount(tasks):
    for task in tasks.values():
        reviewer, outcome = task.evaluator.reviewer_behavior, task.evaluator.evaluation.outcome
        if reviewer and reviewer.decision == "approve":
            assert outcome.refund_total_minor == reviewer.applies_to.amount_minor, task.id
        elif outcome.refund_total_minor:
            pytest.fail(f"{task.id} expects a refund without an approving reviewer")
