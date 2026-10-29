"""Export every Mermaid diagram of hld.md into docs/diagrams/ (W5.10, §16).

    uv run python -m scripts.export_diagrams             # write the .mmd sources
    uv run python -m scripts.export_diagrams --render    # also render SVGs (needs Node)

`hld.md` stays the single source: each diagram is written to `NN-<section>.mmd` named after the
heading above it, and `docs/diagrams/README.md` indexes them. Rendering uses mermaid-cli
(`npx @mermaid-js/mermaid-cli`) with the Chrome or Edge already on the machine, so nothing is
downloaded for a browser. Run it again after editing hld.md.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "hld.md"
OUT = ROOT / "docs" / "diagrams"
MERMAID_CLI = "@mermaid-js/mermaid-cli@11.4.2"  # a pinned release
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


@dataclass
class Diagram:
    number: int
    heading: str
    source: str

    @property
    def stem(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.heading.lower()).strip("-")
        return f"{self.number:02d}-{slug}"[:60].rstrip("-")


def extract(text: str) -> list[Diagram]:
    """Each ```mermaid block, named by the nearest heading above it."""
    diagrams: list[Diagram] = []
    heading = "diagram"
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        elif line.strip() == "```mermaid":
            block: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                block.append(lines[i])
                i += 1
            diagrams.append(Diagram(len(diagrams) + 1, heading, "\n".join(block) + "\n"))
        i += 1
    return diagrams


def browser() -> str | None:
    return next((path for path in BROWSERS if Path(path).exists()), None)


def render(diagram: Diagram, executable: str) -> None:
    source, target = OUT / f"{diagram.stem}.mmd", OUT / f"{diagram.stem}.svg"
    config = OUT / ".puppeteer.json"
    config.write_text(json.dumps({"executablePath": executable}), encoding="utf-8")
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        raise SystemExit("npx (Node.js) is needed to render; the .mmd files were written")
    command = [npx, "--yes", MERMAID_CLI, "-i", str(source), "-o", str(target), "-p", str(config)]
    environment = {**os.environ, "PUPPETEER_SKIP_DOWNLOAD": "1"}
    subprocess.run(command, check=True, cwd=ROOT, env=environment)
    config.unlink()


def index(diagrams: list[Diagram], rendered: bool) -> str:
    rows = [
        "# Architecture diagrams",
        "",
        (
            "Exported from [`hld.md`](../../hld.md), which stays the source. Regenerate with "
            "`uv run python -m scripts.export_diagrams --render`. GitHub also renders the "
            "diagrams inside `hld.md` directly."
        ),
        "",
        "| # | Diagram | Source | Image |",
        "|---|---|---|---|",
    ]
    for d in diagrams:
        image = f"[svg]({d.stem}.svg)" if rendered else "run with `--render`"
        rows.append(f"| {d.number} | {d.heading} | [`{d.stem}.mmd`]({d.stem}.mmd) | {image} |")
    rows.append("")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true", help="also render SVGs with mermaid-cli")
    args = parser.parse_args()
    diagrams = extract(SOURCE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.mmd"):
        old.unlink()
    for diagram in diagrams:
        (OUT / f"{diagram.stem}.mmd").write_text(diagram.source, encoding="utf-8", newline="\n")
    print(f"wrote {len(diagrams)} diagram sources to {OUT.relative_to(ROOT)}")
    rendered = False
    if args.render:
        executable = browser()
        if executable is None:
            raise SystemExit("no Chrome or Edge found; install one or edit BROWSERS")
        for diagram in diagrams:
            print(f"rendering {diagram.stem}", flush=True)
            render(diagram, executable)
        rendered = True
    (OUT / "README.md").write_text(index(diagrams, rendered), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
