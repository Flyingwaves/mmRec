"""
Configuration loader for mmRec.
"""

import os
import copy
import yaml
import argparse
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# DotDict: dict subclass that supports attribute-style access
# ═══════════════════════════════════════════════════════════════════════════════

class DotDict(dict):
    """
    A dictionary that supports attribute-style access.
    Nested dictionaries are also converted to DotDicts recursively.
    """
    def __init__(self, d: dict = None):
        super().__init__()
        if d:
            for k, v in d.items():
                self[k] = v
    
    def __setitem__(self, key, value: Any):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'.")
    
    def __setattr__(self, key, value: Any):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'.")
        
    def __repr__(self):
        lines = []
        for k, v in self.items():
            if isinstance(v, DotDict):
                for sub_k, sub_v in v.items():
                    lines.append(f"  {k}.{sub_k} = {sub_v!r}")
            else:
                lines.append(f"{k} = {v!r}")
        return "Config(\n" + "\n".join(lines) + "\n)"


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: '{file_path}'")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f) or {}
    

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries. Values in 'override' take precedence over 'base'.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _cast(value: str) -> Any:
    """
    Best-effort type casting for --set values coming in as strings.
    """
    if value.lower() in ("true"):
        return True
    if value.lower() in ("false"):
        return False
    if value.lower() in ("none", "null"):
        return None
    
    try:
        parsed = yaml.safe_load(value)
        if not isinstance(parsed, str):
            return parsed
    except yaml.YAMLError:
        pass
    return value


def _apply_overrides(config: dict, overrides: list) -> dict:
    """
    Apply command-line overrides to the config dictionary.

    Overrides are in the form of "key=value", nested keys use dot notation,
    e.g. "model.hidden_size=1024"

    If an intermediate key is missing or is not a dict, it is replaced with 
    an empty dict so nested values can be assigned.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set value must be in the form 'key=value', got '{item}'")
        key_path, _, raw_value = item.partition("=")
        keys = key_path.strip().split(".")
        value = _cast(raw_value.strip())

        node = config
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return config

# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_config(args: argparse.Namespace) -> DotDict:
    """
    Build a merged DotDict config from:
        1. configs/train.yaml
        2. configs/models/{model}.yaml
        3. --set overrides from command line
    """
    train_yaml_path = os.path.join(_REPO_ROOT, "configs", "train.yaml")
    model_yaml_path = os.path.join(_REPO_ROOT, "configs", "models", f"{args.model}.yaml")

    train_raw = _load_yaml(train_yaml_path)
    model_raw = _load_yaml(model_yaml_path)

    merged = _deep_merge(base=train_raw, override=model_raw)

    merged.setdefault("meta", {})
    merged["meta"]["model"] = args.model
    merged["meta"]["dataset"] = args.dataset

    if getattr(args, "set", None):
        merged = _apply_overrides(merged, args.set)

    return DotDict(merged)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add config-related arguments to the parser.
    Call this in scripts/train.py and scripts/infer.py.
    """
    parser.add_argument("--dataset", type=str, default="taac2026",
                        help="Dataset name (must match configs/{dataset}/ folder)")
    parser.add_argument("--model", type=str, default="onetrans",
                        help="Model name (must match configs/models/{model}.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values using dot notation, e.g. --set train.lr_dense=1e-4 model.num_layers=2")
    
    return parser