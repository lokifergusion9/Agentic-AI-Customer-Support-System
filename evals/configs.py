"""Register agent configurations and prompt versions in the `eval` schema (W4.7, W2.16, E§22).

An `agent_configs` row is immutable and named; `config_hash` covers everything that changes
behaviour (adapter, model, prompts by content, architecture, limits, tool schemas), so the same
settings never get two names and a name never means two settings. Registering again with the
same name and the same settings is a no-op; with different settings it is refused.
"""

import hashlib
import json
from typing import Any

from sqlalchemy import Connection, text

from app.agents.config import AgentConfig
from app.tools.registry import MODEL_TOOLS, TOOLS
from app.workflows.runner import PROMPTS
from evals.db import EVAL_SCHEMA

LIMIT_FIELDS = ("temperature", "max_output_tokens", "max_steps", "max_tokens", "max_retries")


def sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def tool_schema_hash() -> str:
    """Hash of every tool the model may be offered: names, descriptions and argument schemas."""
    return sha256(
        {name: [TOOLS[name].description, TOOLS[name].json_schema()] for name in MODEL_TOOLS}
    )


def config_row(config_id: str, config: AgentConfig, adapter: str = "langgraph") -> dict[str, Any]:
    prompts = {name: config.prompt(name).sha256 for name in PROMPTS.get(config.architecture, [])}
    row = {
        "adapter": adapter,
        "provider": config.provider,
        "model_id": config.model_id,
        "prompt_version": config.prompt_version,
        "architecture": config.architecture,
        "prompt_hashes": prompts,
        "limits": {name: getattr(config, name) for name in LIMIT_FIELDS},
        "tool_schema_hash": tool_schema_hash(),
    }
    return {"id": config_id, **row, "config_hash": sha256(row)}


def agent_config(row: dict[str, Any]) -> AgentConfig:
    """The workflow's AgentConfig from an `agent_configs` row."""
    return AgentConfig(
        provider=row["provider"],
        model_id=row["model_id"],
        prompt_version=row["prompt_version"],
        architecture=row["architecture"],
        **{name: row["limits"][name] for name in LIMIT_FIELDS if name in row["limits"]},
    )


def register_agent_config(connection: Connection, row: dict[str, Any]) -> dict[str, Any]:
    """Store the row (and its prompts) unless it is already there; refuse a changed one."""
    existing = connection.execute(
        text(f"SELECT id, config_hash FROM {EVAL_SCHEMA}.agent_configs WHERE id = :id"),
        {"id": row["id"]},
    ).one_or_none()
    if existing is not None:
        if existing.config_hash != row["config_hash"]:
            raise ValueError(
                f"agent config {row['id']!r} is registered with other settings; "
                "agent configs are immutable, register a new name"
            )
        return row
    if row["adapter"] == "langgraph":
        register_prompts(connection, agent_config(row))
    connection.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.agent_configs (id, adapter, provider, model_id, "
            "prompt_version, architecture, prompt_hashes, limits, tool_schema_hash, config_hash) "
            "VALUES (:id, :adapter, :provider, :model_id, :prompt_version, :architecture, "
            "CAST(:prompt_hashes AS jsonb), CAST(:limits AS jsonb), :tool_schema_hash, "
            ":config_hash)"
        ),
        {
            **row,
            "prompt_hashes": json.dumps(row["prompt_hashes"]),
            "limits": json.dumps(row["limits"]),
        },
    )
    return row


def register_prompts(connection: Connection, config: AgentConfig) -> None:
    for name in PROMPTS[config.architecture]:
        prompt = config.prompt(name)
        stored = connection.execute(
            text(f"SELECT sha256 FROM {EVAL_SCHEMA}.prompt_versions WHERE id = :id"),
            {"id": prompt.label},
        ).scalar_one_or_none()
        if stored is None:
            connection.execute(
                text(
                    f"INSERT INTO {EVAL_SCHEMA}.prompt_versions (id, name, version, sha256, "
                    "content) VALUES (:id, :name, :version, :sha256, :content)"
                ),
                {
                    "id": prompt.label,
                    "name": prompt.name,
                    "version": prompt.version,
                    "sha256": prompt.sha256,
                    "content": prompt.text,
                },
            )
        elif stored != prompt.sha256:
            raise ValueError(
                f"prompt {prompt.label} changed since it was registered; "
                "save the change as a new prompt version"
            )


def scripted_config_row(script: str) -> dict[str, Any]:
    """The scripted adapter as an agent config: the script name stands in for the model."""
    row = {
        "adapter": "scripted",
        "provider": "none",
        "model_id": script,
        "prompt_version": "none",
        "architecture": "scripted",
        "prompt_hashes": {},
        "limits": {},
        "tool_schema_hash": tool_schema_hash(),
    }
    return {"id": f"scripted_{script}", **row, "config_hash": sha256(row)}


def named_config(name: str) -> dict[str, Any]:
    """`scripted_<script>`, or `settings`: the agent the .env configures (AgentConfig)."""
    if name.startswith("scripted_"):
        return scripted_config_row(name.removeprefix("scripted_"))
    if name == "settings":
        config = AgentConfig.from_settings()
        return config_row(config.config_id, config)
    raise ValueError(f"unknown config {name!r}: use scripted_<script> or settings")
