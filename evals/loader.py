"""Load evaluation tasks and split them into the agent and evaluator views (W4.1).

    uv run python -m evals.loader support-v1          # validate every task against the lock
    uv run python -m evals.loader support-v1 --lock   # record new task versions in the lock

A task version is immutable: its content hash is recorded in `<dataset>/task_versions.json`.
Changing a task means bumping its `version` and locking again; changing it in place (same
version, different hash) is refused, so scores stored for an old version keep meaning what they
meant. The hash is over the parsed task, so reformatting the YAML or editing comments is not a
change.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.task_format import AgentView, EvaluatorView, Task

DATASETS = Path(__file__).resolve().parent / "datasets"
LOCK_FILE = "task_versions.json"


class TaskVersionError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedTask:
    agent: AgentView
    evaluator: EvaluatorView

    @property
    def id(self) -> str:
        return self.agent.task_id


def content_hash(task: Task) -> str:
    canonical = json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def split_views(task: Task) -> LoadedTask:
    rest = task.model_dump(exclude={"id", "agent_input"})
    return LoadedTask(
        agent=AgentView(task_id=task.id, agent_input=task.agent_input),
        evaluator=EvaluatorView(task_id=task.id, content_hash=content_hash(task), **rest),
    )


def parse_task(path: Path) -> Task:
    task = Task.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if path.stem != task.id:
        raise ValueError(f"{path.name}: file name must be the task id {task.id!r}")
    return task


def read_lock(dataset_dir: Path) -> dict[str, dict[str, str]]:
    path = dataset_dir / LOCK_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_version(task: Task, lock: dict[str, dict[str, str]]) -> bool:
    """True if this version is already locked; raises if it was edited in place or went back."""
    versions = lock.get(task.id, {})
    locked = versions.get(str(task.version))
    if locked is not None and locked != content_hash(task):
        raise TaskVersionError(
            f"{task.id} v{task.version} changed after it was locked; bump its version instead"
        )
    if versions and task.version < max(map(int, versions)):
        raise TaskVersionError(f"{task.id} v{task.version} is older than a locked version")
    return locked is not None


def load_dataset(dataset: str, *, require_locked: bool = True) -> list[LoadedTask]:
    dataset_dir = DATASETS / dataset
    lock = read_lock(dataset_dir)
    loaded = []
    for path in sorted((dataset_dir / "tasks").glob("*.yaml")):
        task = parse_task(path)
        if not check_version(task, lock) and require_locked:
            raise TaskVersionError(f"{task.id} v{task.version} is not locked; run with --lock")
        loaded.append(split_views(task))
    return loaded


def split_ids(dataset: str, split: str) -> list[str]:
    """Task IDs listed in `<dataset>/splits/<split>.txt`, skipping comments.

    `all` is every task of the dataset (dev, validation and test together).
    """
    if split == "all":
        return sorted(path.stem for path in (DATASETS / dataset / "tasks").glob("*.yaml"))
    path = DATASETS / dataset / "splits" / f"{split}.txt"
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def load_split(dataset: str, split: str) -> list[LoadedTask]:
    wanted = split_ids(dataset, split)
    by_id = {task.id: task for task in load_dataset(dataset)}
    if missing := [task_id for task_id in wanted if task_id not in by_id]:
        raise ValueError(f"{split} lists unknown tasks: {missing}")
    return [by_id[task_id] for task_id in wanted]


def lock_dataset(dataset: str) -> list[str]:
    """Record the hash of every unlocked task version. Locked versions are never rewritten."""
    dataset_dir = DATASETS / dataset
    lock = read_lock(dataset_dir)
    added = []
    for path in sorted((dataset_dir / "tasks").glob("*.yaml")):
        task = parse_task(path)
        if not check_version(task, lock):
            lock.setdefault(task.id, {})[str(task.version)] = content_hash(task)
            added.append(f"{task.id} v{task.version}")
    (dataset_dir / LOCK_FILE).write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("--lock", action="store_true", help="record new task versions")
    args = parser.parse_args()
    if args.lock:
        added = lock_dataset(args.dataset)
        print(f"locked {len(added)} new task version(s)", *added, sep="\n  ")
    tasks = load_dataset(args.dataset)
    print(f"{args.dataset}: {len(tasks)} task(s) valid and locked")


if __name__ == "__main__":
    main()
