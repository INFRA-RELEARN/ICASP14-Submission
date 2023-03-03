
from .registry import get_registry_entry

from .systems import gamma_process


def get_env(hyper_dict):
    return get_registry_entry(hyper_dict['env_id']).cls(**hyper_dict['env_kwargs'])
