"""Parsing of the policy documents into sections and citations (W2.1)."""

from datetime import date

import pytest

from app.policies.documents import load_policies, parse_policy

VALID = """---
id: sample_policy
version: v2
effective_from: 2026-01-01
effective_to: null
title: Sample Policy
---

## Section 1: Scope

Covers samples.

## Section 2: Rules

- one
- two
"""


def test_every_shipped_document_parses() -> None:
    documents = load_policies()
    assert [d.document_id for d in documents] == [
        "damaged_products_v1",
        "human_review_exceptions_v1",
        "refund_policy_v0",
        "refund_policy_v1",
        "return_processing_v1",
        "shipping_delays_v1",
    ]
    assert all(len(d.sections) >= 6 for d in documents)


def test_refund_policy_v1_section_3_is_the_eligibility_rule() -> None:
    refund = next(d for d in load_policies() if d.document_id == "refund_policy_v1")
    section = refund.sections[2]
    assert refund.citation(section.number) == "refund_policy_v1:section_3"
    assert section.title == "Eligibility"
    assert "inspection passes" in section.content
    assert refund.effective_from == date(2026, 1, 1)
    assert refund.effective_to is None


def test_refund_policy_versions_do_not_overlap() -> None:
    v0, v1 = (d for d in load_policies() if d.policy_key == "refund_policy")
    assert v0.effective_to == date(2025, 12, 31)
    assert v0.effective_to < v1.effective_from


def test_parse_splits_sections_and_hashes_content() -> None:
    document = parse_policy(VALID, "sample.md")
    assert document.document_id == "sample_policy_v2"
    assert [(s.number, s.title) for s in document.sections] == [(1, "Scope"), (2, "Rules")]
    assert document.sections[1].content == "- one\n- two"
    assert parse_policy(VALID.replace("\n", "\r\n"), "x").content_hash == document.content_hash
    assert parse_policy(VALID + "more", "x").content_hash != document.content_hash


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("## Section 1: Scope\n\nNo front matter.\n", "missing front matter"),
        (VALID.replace("title: Sample Policy\n", ""), "missing title"),
        (VALID.replace("## Section 2", "## Section 3"), "out of order"),
        (VALID.replace("Covers samples.", ""), "section 1 is empty"),
        (VALID.split("## Section 1")[0], "no '## Section N: Title' headings"),
    ],
)
def test_parse_rejects_malformed_documents(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_policy(text, "bad.md")
