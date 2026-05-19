import torch
from contextlib import contextmanager

def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    return torch.amp.autocast(
        device_type=device_type,
        enabled=enabled,
        dtype=dtype,
        cache_enabled=cache_enabled
    )
