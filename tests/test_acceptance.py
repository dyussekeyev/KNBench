"""Repository-wide acceptance checks (TECHSPEC.md §18, EXECPLAN T7.7)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_NAME = "KazNa" + "tSecBench"
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
SKIP_PARTS = {".venv", ".git", "__pycache__", ".freebuff", ".mypy_cache", ".pytest_cache"}


def _tree(pattern: str) -> list[Path]:
    return [
        path
        for path in ROOT.rglob(pattern)
        if path.is_file() and not SKIP_PARTS.intersection(path.parts)
    ]


def test_no_print_in_library_modules() -> None:
    offenders = [
        str(path)
        for path in (ROOT / "knbench").rglob("*.py")
        if "print(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_legacy_project_name_is_absent() -> None:
    offenders = []
    for pattern in ("*.md", "*.py", "*.json", "*.ini", "*.txt", "*.example"):
        for path in _tree(pattern):
            if LEGACY_NAME in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
    assert offenders == []


def test_library_code_contains_no_cyrillic() -> None:
    offenders = [
        str(path)
        for path in (ROOT / "knbench").rglob("*.py")
        if CYRILLIC.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def _headings(path: Path) -> list[int]:
    levels: list[int] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+", line)
        if match:
            levels.append(len(match.group(1)))
    return levels


def test_readmes_share_the_same_section_structure() -> None:
    english = _headings(ROOT / "README.md")
    kazakh = _headings(ROOT / "README.kk.md")
    russian = _headings(ROOT / "README.ru.md")
    assert english[0] == 1
    assert english == kazakh == russian
    for name in ("README.md", "README.kk.md", "README.ru.md"):
        first_line = (ROOT / name).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == "# KNBench"


def test_dataset_fixtures_contain_no_credentials() -> None:
    for path in _tree("*.json"):
        if "data" not in path.parts and "fixtures" not in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "sk-or-v1-" not in text, path
        assert "OPENROUTER_API_KEY=" not in text, path


DOCS = ROOT / "docs"
USER_GUIDES = ("user-guide.md", "user-guide.kk.md", "user-guide.ru.md")
TECHNICAL_DOCS = ("dataset-format.md", "methodology.md", "configuration.md")


def test_documentation_deliverables_exist() -> None:
    for name in USER_GUIDES + TECHNICAL_DOCS:
        assert (DOCS / name).is_file(), name


def test_license_files_exist() -> None:
    apache = (ROOT / "LICENSE").read_text(encoding="utf-8")
    data = (ROOT / "LICENSE-DATA").read_text(encoding="utf-8")
    assert "Apache License" in apache
    assert "Version 2.0" in apache
    assert "CC BY 4.0" in data
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in pyproject


def test_user_guides_share_the_same_section_structure() -> None:
    structures = [_headings(DOCS / name) for name in USER_GUIDES]
    assert structures[0][0] == 1
    assert structures[0] == structures[1] == structures[2]


def test_english_documentation_has_no_cyrillic() -> None:
    offenders = [
        name
        for name in USER_GUIDES[:1] + TECHNICAL_DOCS
        if CYRILLIC.search((DOCS / name).read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_documentation_links_resolve() -> None:
    link = re.compile(r"\]\(([^)#]+\.md)\)")
    missing: list[tuple[str, str]] = []
    for name in USER_GUIDES + TECHNICAL_DOCS:
        text = (DOCS / name).read_text(encoding="utf-8")
        for target in link.findall(text):
            if not (DOCS / target).is_file():
                missing.append((name, target))
    assert missing == []


def test_user_guides_link_to_every_language() -> None:
    expected = {
        "user-guide.md": ("user-guide.kk.md", "user-guide.ru.md"),
        "user-guide.kk.md": ("user-guide.md", "user-guide.ru.md"),
        "user-guide.ru.md": ("user-guide.md", "user-guide.kk.md"),
    }
    for name, targets in expected.items():
        text = (DOCS / name).read_text(encoding="utf-8")
        for target in targets:
            assert f"({target})" in text, (name, target)
