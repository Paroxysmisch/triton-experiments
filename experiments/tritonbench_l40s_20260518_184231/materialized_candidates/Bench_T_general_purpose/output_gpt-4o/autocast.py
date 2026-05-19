import torch
from contextlib import contextmanager

@contextmanager
def autocast(device_type='cuda', enabled=True, dtype=None, cache_enabled=True):
    """
    Triton wrapper for mixed precision autocast.

    Parameters:
    - device_type (str): The type of device to run on, default is 'cuda'.
    - enabled (bool): Whether autocasting is enabled, default is True.
    - dtype (torch.dtype or None): The data type to cast to, default is None (automatic selection).
    - cache_enabled (bool): Whether to use a cache for storing computation graphs, default is True.

    Returns:
    - ContextManager: A context manager that manages the autocast state.
    """
    if device_type != 'cuda':
        raise ValueError("autocast is only supported for 'cuda' device type.")
    
    # Set the autocast state
    with torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled) as ac:
        try:
            yield ac
        finally:
            # Any necessary cleanup can be done here
            pass

# Example usage
def forward_pass(model, input):
    with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
        output = model(input)
        loss = torch.nn.functional.mse_loss(output, target)
    return loss
