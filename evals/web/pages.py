"""Browser screens for the evaluation platform (W5.6): experiments, one experiment, one run, and
a comparison. Like the reviewer screen, the pages carry no data: the script in each page calls
the operator-only /eval API with the token the user pastes in, so every read goes through the
same authorization as any other client.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import app.web.pages as agent_pages

# The evaluation templates extend the agent app's base template.
AGENT_TEMPLATES = Path(agent_pages.__file__).parent / "templates"
templates = Jinja2Templates(directory=[Path(__file__).parent / "templates", AGENT_TEMPLATES])

router = APIRouter(include_in_schema=False)


@router.get("/ui/experiments", response_class=HTMLResponse)
def experiments(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "experiments.html")


@router.get("/ui/experiments/{experiment_id}", response_class=HTMLResponse)
def experiment(request: Request, experiment_id: str) -> HTMLResponse:
    return templates.TemplateResponse(request, "experiment.html", {"experiment_id": experiment_id})


@router.get("/ui/runs/{run_id}", response_class=HTMLResponse)
def run(request: Request, run_id: str) -> HTMLResponse:
    return templates.TemplateResponse(request, "run.html", {"run_id": run_id})


@router.get("/ui/compare", response_class=HTMLResponse)
def compare(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "compare.html")
