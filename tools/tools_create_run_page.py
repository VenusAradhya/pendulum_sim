#!/usr/bin/env python3
"""Create one flat docs/runs/run_XXX.md archive page per pipeline run.

This wrapper intentionally delegates to tools_archive_run so both:
- `python tools/tools_archive_run.py`
- `python tools/tools_create_run_page.py`

behave the same way and generate the same run-page format.
"""

from tools_archive_run import create_run_page, ROOT


def main() -> None:
    page = create_run_page()
    print(f"Run page created: {page.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
