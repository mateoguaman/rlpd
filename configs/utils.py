from typing import Any, Optional, Dict
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, fields, field, asdict, is_dataclass


def flatten_config(cfg: Any, prefix: Optional[str] = '') -> Dict[str, Any]:
    flat_config = {}
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if prefix is None:
                new_prefix = None
                flat_key = key
            else:
                new_prefix = f"{prefix}{key}." if prefix else f"{key}."
                flat_key = new_prefix[:-1]
            
            if isinstance(value, DictConfig):
                flat_config.update(flatten_config(value, new_prefix))
            else:
                flat_config[flat_key] = value
    return flat_config

def to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

def get_flat_config(cfg: DictConfig, use_prefix: bool = True) -> Dict[str, Any]:
    return flatten_config(cfg, '' if use_prefix else None)


def flatten_config_dataclass(obj: Any, prefix: Optional[str] = '') -> Dict[str, Any]:
    flat_config = {}
    if is_dataclass(obj):
        for field in fields(obj):
            value = getattr(obj, field.name)
            if prefix is None:
                new_prefix = None
                key = field.name
            else:
                new_prefix = f"{prefix}{field.name}." if prefix else f"{field.name}."
                key = new_prefix[:-1]
            
            if is_dataclass(value):
                flat_config.update(flatten_config(value, new_prefix))
            else:
                flat_config[key] = value
    return flat_config