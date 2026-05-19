import torch
from contextlib import contextmanager

@contextmanager
def autocast(device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
    """
    A context manager for running operations in mixed precision.
    
    Args:
        device_type (str): The type of device, e.g., "cuda".
        enabled (bool): Whether to enable autocasting.
        dtype (torch.dtype): The data type to cast to. If None, autodetect.
        cache_enabled (bool): Whether to enable caching of autocast states.
        
    Yields:
        None
    """
    # Create the autocast context manager
    autocast_context = torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
    
    # Yield the context manager
    yield autocast_context
