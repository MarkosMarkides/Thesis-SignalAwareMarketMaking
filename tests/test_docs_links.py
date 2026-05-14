import re
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files(project_root):
    files = [project_root / "README.md", project_root / "methodology_focused_thesis_draft.md"]
    files.extend(sorted((project_root / "docs").glob("*.md")))
    files.extend(sorted((project_root / "docs" / "python-files").glob("*.md")))
    return [path for path in files if path.exists()]


def _local_link_target(path, raw_link):
    link = raw_link.strip()
    if link.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if link.startswith("<") and link.endswith(">"):
        link = link[1:-1]
    link = link.split("#", 1)[0]
    if not link:
        return None
    return (path.parent / unquote(link)).resolve()


def test_markdown_local_links_point_to_existing_files(project_root):
    missing = []
    for path in _markdown_files(project_root):
        for raw_link in MARKDOWN_LINK.findall(path.read_text()):
            target = _local_link_target(path, raw_link)
            if target is not None and not target.exists():
                missing.append(f"{path.relative_to(project_root)} -> {raw_link}")

    assert missing == []


def test_docs_reference_current_source_paths(project_root):
    text = "\n".join(path.read_text() for path in _markdown_files(project_root))

    assert "technical/labelling.py" in text
    assert "technical/backtesting.py" in text


def test_docs_do_not_describe_deleted_source_paths_as_active_files(project_root):
    text = "\n".join(path.read_text() for path in _markdown_files(project_root))

    forbidden = [
        "feature_engineering/labelling.py",
        "backtesting/backtesting.py",
        "main.py",
        "strategies/stoikov_calibration.py",
    ]

    for deleted_path in forbidden:
        assert deleted_path not in text

