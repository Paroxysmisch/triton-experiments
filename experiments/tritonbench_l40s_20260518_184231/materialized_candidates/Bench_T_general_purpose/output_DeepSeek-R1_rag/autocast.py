import torch
from contextlib import contextmanager

def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    """
    Creates a context manager for automatic mixed precision (AMP) training.
    
    Args:
        device_type (str): Target device type (e.g., "cuda").
        enabled (bool): Whether autocasting should be enabled.
        dtype (torch.dtype, optional): Override for the target dtype (default: None).
        cache_enabled (bool): Enables the weight cache in autocast.
    
    Returns:
        ContextManager: Configured autocast context manager.
    """
    return torch.amp.autocast(device_type=device_type, 
                             enabled=enabled, 
                             dtype=dtype, 
                             cache_enabled=cache_enabled)
