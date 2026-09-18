"""[pkg] Run configuration. Reads ./config.yaml (the run folder's snapshot) when present, else the package's
config.yaml. `get("writer.model")` walks the dotted path; a missing key returns `default`."""

import os

import yaml

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
_cache = None


def _load():
    global _cache
    if _cache is None:
        path = "config.yaml" if os.path.exists("config.yaml") else _PKG
        with open(path, encoding="utf-8") as f:
            _cache = yaml.safe_load(f) or {}
        # a run snapshot may be partial: fall back to the package defaults key by key
        with open(_PKG, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        for k, v in base.items():
            if isinstance(v, dict):
                _cache[k] = {**v, **(_cache.get(k) or {})}
            else:
                _cache.setdefault(k, v)
    return _cache


def get(path, default=None):
    cur = _load()
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
