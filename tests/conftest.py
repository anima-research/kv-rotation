"""Shared test config: make the local anamnesis-pl checkout importable (exp11 bridge).

anamnesis-pl is a read-only PYTHONPATH dependency, never installed as a package
(see src/kvrot/sigbridge.py). Tests that import it skip cleanly when no checkout
is available (e.g. CI boxes without the research tree).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ANAMNESIS_CANDIDATES = (
    "~/projects/anamnesis_exps/pipeline",   # dev box
    "~/luxi-files/anamnesis-pl-exp11",      # node1 staging copy
)


def _ensure_anamnesis_on_path() -> None:
    try:
        import anamnesis  # noqa: F401

        return
    except ImportError:
        pass
    for cand in _ANAMNESIS_CANDIDATES:
        p = Path(cand).expanduser()
        if (p / "anamnesis" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            return


_ensure_anamnesis_on_path()
