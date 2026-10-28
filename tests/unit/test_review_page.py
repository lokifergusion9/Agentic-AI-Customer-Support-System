"""The reviewer screen (W3.8): served without data, and in step with the queue's fields."""

import re

from fastapi.testclient import TestClient

from app.main import app
from app.services.approvals import PendingApprovalView
from app.web.pages import templates

client = TestClient(app)


def page() -> str:
    response = client.get("/review")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    return response.text


def test_page_has_the_review_controls() -> None:
    html = page()
    for control in ("Approve", "Reject", "Show timeline", "Reviewer token"):
        assert control in html
    # The page reads and decides through the authorized JSON API, not server-side data.
    assert 'api("GET", "/approvals")' in html
    assert "/decision" in html


def test_page_shows_every_field_the_screen_needs() -> None:
    """W3.8: customer and order, complaint, action and amount, evidence, citations, expiry."""
    source = templates.env.loader.get_source(templates.env, "review.html")[0]
    used = set(re.findall(r"approval\.(\w+)", source))
    assert used <= set(PendingApprovalView.model_fields)
    required = {
        "customer_name",
        "order_id",
        "complaint_summary",
        "action",
        "amount_minor",
        "evidence_refs",
        "policy_refs",
        "expires_at",
    }
    assert required <= used


def test_model_text_is_never_inserted_as_html() -> None:
    assert "innerHTML" not in page()
