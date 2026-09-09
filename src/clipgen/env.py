"""Load environment variables from the project's git-ignored ``.env``.

Single-user, local tool: secrets (e.g. ``GEMINI_API_KEY``) and the model paths live in a
``.env`` at the repo root so the web server and ad-hoc scripts all see the same values without
exporting them in every shell. Loading is idempotent and never overrides a variable already set
in the real environment (an explicit ``export`` still wins).
"""

from __future__ import annotations

from pathlib import Path

_loaded = False


def load_env() -> None:
    """Load the repo-root ``.env`` once (no-op if already loaded or the file is absent)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from dotenv import load_dotenv

    # env.py lives at <repo>/src/clipgen/env.py → parents[2] is the repo root.
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()  # fall back to a .env found by walking up from the cwd
