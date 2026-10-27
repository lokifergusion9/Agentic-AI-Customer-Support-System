"""One agent configuration object and versioned prompts (W2.16, E§22).

Model, prompt version, architecture and limits come from here and nowhere else, so the
evaluation platform can register each combination as an immutable `agent_configs` row (W4.3)
and compare them (W5.2). Every run records the configuration and the hash of each prompt used.
"""

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from app.config import Settings, get_settings
from app.llm.gemini import GeminiProvider
from app.llm.types import ModelProvider

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
Architecture = Literal["single", "specialist"]


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.name}_{self.version}"


@cache
def load_prompt(name: str, version: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}_{version}.md"
    return Prompt(name, version, path.read_text(encoding="utf-8").replace("\r\n", "\n").strip())


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model_id: str
    prompt_version: str
    architecture: Architecture
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_steps: int = 25
    max_tokens: int = 50000
    max_retries: int = 2

    @property
    def config_id(self) -> str:
        """Matches the W5.2 naming, e.g. gemini-2.x-flash_specialist_v1."""
        return f"{self.model_id or 'unset'}_{self.architecture}_{self.prompt_version}"

    def prompt(self, name: str) -> Prompt:
        return load_prompt(name, self.prompt_version)

    def summary(self, prompt_names: list[str]) -> dict[str, object]:
        """Recorded with each run (W2.16): what exactly produced this output."""
        return {
            "config_id": self.config_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "prompt_version": self.prompt_version,
            "prompt_sha256": {name: self.prompt(name).sha256[:16] for name in prompt_names},
            "limits": {"max_steps": self.max_steps, "max_tokens": self.max_tokens},
        }

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AgentConfig":
        settings = settings or get_settings()
        if settings.agent_architecture not in ("single", "specialist"):
            raise ValueError(f"unknown AGENT_ARCHITECTURE {settings.agent_architecture!r}")
        return cls(
            provider=settings.llm_provider,
            model_id=settings.llm_model,
            prompt_version=settings.agent_prompt_version,
            architecture=settings.agent_architecture,  # type: ignore[arg-type]
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            max_steps=settings.max_workflow_steps,
            max_tokens=settings.max_run_tokens,
            max_retries=settings.llm_max_retries,
        )


def build_provider(config: AgentConfig, settings: Settings | None = None) -> ModelProvider:
    settings = settings or get_settings()
    if config.provider != "gemini":
        raise ValueError(f"unsupported LLM_PROVIDER {config.provider!r}")
    return GeminiProvider(
        config.model_id,
        settings.llm_api_key.get_secret_value(),
        timeout_seconds=settings.llm_timeout_seconds,
    )
