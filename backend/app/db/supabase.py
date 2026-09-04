"""
backend/app/db/supabase.py  (SHIM — do not import directly)
=============================================================
#17: The actual DataStore implementation has moved to db/datastore.py.
     This file is kept as a backward-compatibility re-export so any cached
     imports or __pycache__ references continue to work during the transition.
     It will be removed in a future cleanup pass once all imports are updated.
"""
from .datastore import DataStore  # noqa: F401 — re-export

__all__ = ["DataStore"]
