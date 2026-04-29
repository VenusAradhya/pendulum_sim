"""Thin RL entrypoint.

Why this file exists:
- Keep user-facing CLI/import path stable (`pendulum_sim.rl_pipeline:main`).
- Keep orchestration code short and readable.
- Delegate implementation details to `pendulum_sim.rl_core`.
"""

from pendulum_sim.rl_core import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
