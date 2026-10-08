"""Experiment configuration: composition, validation and checkpoint migration.

``loader`` composes fresh runs (``config/default.yaml`` + one model variant from
``config/variants/`` + one data profile from ``config/data/`` + CLI) and restores
checkpoint-embedded configs; ``definitions`` holds the variant registry and
protected keys, ``validation`` the per-variant contracts, ``migration`` the
upgrades of older checkpoint configs. The loader's public API is re-exported here.
"""
from .loader import *  # noqa: F401,F403
from .loader import __all__  # noqa: F401
