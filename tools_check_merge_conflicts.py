#!/usr/bin/env python3
"""Fail fast when git conflict markers are present in source files."""
from pathlib import Path
import sys

MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
INCLUDE_SUFFIXES = {".py", ".rst", ".md", ".txt", ".yaml", ".yml", ".toml", ".json"}
IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", "docs/_build"}


def should_scan(path: Path) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in INCLUDE_SUFFIXES


def main() -> int:
    root = Path.cwd()
    hits: list[tuple[Path, int, str]] = []

    for path in root.rglob("*"):
        if not path.is_file() or not should_scan(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if any(stripped.startswith(marker) for marker in MARKERS):
                # Ignore rst underline lines made of '='
                if stripped and set(stripped) == {"="}:
                    continue
                hits.append((path.relative_to(root), idx, stripped[:120]))

    if not hits:
        print("No merge conflict markers found.")
        return 0

    print("Merge conflict markers found:")
    for path, line_no, snippet in hits:
        print(f"  {path}:{line_no}: {snippet}")
    print("\nResolve markers before running Python or committing.")
    print("Suggested quick recovery for one broken file (replace with branch version):")
    print("  git checkout --theirs <file>   # keep incoming side")
    print("  # or")
    print("  git checkout --ours <file>     # keep current branch side")
    print("  git add <file> && git commit -m 'Resolve merge conflict in <file>'")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
