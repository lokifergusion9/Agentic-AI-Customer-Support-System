"""FastAPI entrypoint: `uv run uvicorn app.main:app --reload`."""

from fastapi import FastAPI

from app.api import approvals, health, runs, tickets


def create_app() -> FastAPI:
    app = FastAPI(title="Customer Support Resolution Agent")
    app.include_router(health.router)
    app.include_router(tickets.router)
    app.include_router(runs.router)
    app.include_router(approvals.router)
    return app


app = create_app()
