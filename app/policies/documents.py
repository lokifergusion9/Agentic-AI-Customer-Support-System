"""Parse the synthetic policy documents in `app/policies/documents/` (W1.1, W2.1).

Each file has a small front matter block (id, version, effective dates, title) and a body split
into `## Section N: Title` headings. One section becomes one chunk, and its citation is
`<policy_key>_<version>:section_<N>`, for example `refund_policy_v1:section_3`.
"""

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"

FRONT_MATTER = re.compile(r"\A---\n(?P<meta>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
SECTION_HEADING = re.compile(r"^## Section (?P<number>\d+): (?P<title>.+)$", re.MULTILINE)
REQUIRED_KEYS = ("id", "version", "effective_from", "effective_to", "title")


@dataclass(frozen=True)
class PolicySection:
    number: int
    title: str
    content: str


@dataclass(frozen=True)
class PolicyDocument:
    policy_key: str
    version: str
    title: str
    effective_from: date
    effective_to: date | None
    source_uri: str
    content_hash: str
    sections: tuple[PolicySection, ...]

    @property
    def document_id(self) -> str:
        return f"{self.policy_key}_{self.version}"

    def citation(self, section_number: int) -> str:
        return f"{self.document_id}:section_{section_number}"


def parse_policy(text: str, source_uri: str) -> PolicyDocument:
    """Parse one policy file. Raises ValueError when the format is wrong."""
    text = text.replace("\r\n", "\n")
    match = FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"{source_uri}: missing front matter")
    meta: dict[str, str] = {}
    for line in match["meta"].splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
    if missing:
        raise ValueError(f"{source_uri}: front matter is missing {', '.join(missing)}")

    body = match["body"]
    headings = list(SECTION_HEADING.finditer(body))
    if not headings:
        raise ValueError(f"{source_uri}: no '## Section N: Title' headings")
    sections = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        number = int(heading["number"])
        if number != index + 1:
            raise ValueError(f"{source_uri}: section {number} is out of order")
        content = body[heading.end() : end].strip()
        if not content:
            raise ValueError(f"{source_uri}: section {number} is empty")
        sections.append(PolicySection(number, heading["title"].strip(), content))

    return PolicyDocument(
        policy_key=meta["id"],
        version=meta["version"],
        title=meta["title"],
        effective_from=date.fromisoformat(meta["effective_from"]),
        effective_to=None
        if meta["effective_to"] == "null"
        else date.fromisoformat(meta["effective_to"]),
        source_uri=source_uri,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        sections=tuple(sections),
    )


def load_policies(directory: Path = DOCUMENTS_DIR) -> list[PolicyDocument]:
    """All policy documents in `directory`, sorted by document ID."""
    documents = [
        parse_policy(path.read_text(encoding="utf-8"), f"app/policies/documents/{path.name}")
        for path in directory.glob("*.md")
    ]
    return sorted(documents, key=lambda document: document.document_id)
