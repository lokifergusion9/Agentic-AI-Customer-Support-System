"""Browser screens (W3.8; W5.6 adds the rest), Jinja2 templates with light JavaScript.

Pages carry no data and need no session: the script in each page calls the JSON API with the
bearer token the user pastes in, so every read and decision goes through the same
authorization as any other API client.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

router = APIRouter(include_in_schema=False)


@router.get("/review", response_class=HTMLResponse)
def review(request: Request) -> HTMLResponse:
    """The reviewer's approval queue: evidence, citations, expiry, approve and reject."""
    return templates.TemplateResponse(request, "review.html")
