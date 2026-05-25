"""OneTrans model — re-exported for the one_trans_eval inference package."""

# one_trans_eval is self-contained: model and dataset are imported from the
# sibling one_trans package so that infer.py can use clean local imports.
import os as _os
import sys as _sys
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_parent_dir = _os.path.dirname(_script_dir)
if _parent_dir not in _sys.path:
    _sys.path.insert(0, _parent_dir)

from one_trans.model import OneTrans  # noqa: E402, F401