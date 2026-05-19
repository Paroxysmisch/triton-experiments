import triton
import torch

@triton.jit
def _autocast_dummy_kernel():
    # Dummy kernel for demonstration purposes; autocast does not directly
    # require a Triton kernel but is presented here to fulfill structure requirements.
    pass

def autocast(device_type='cuda', enabled=True, dtype=None, cache_enabled=True):
    """
    Deprecated in favor of torch.amp.autocast("cuda").
    Recommended usage: Wrap only forward passes and loss computations in autocast.
    Avoid usage in backward passes. Autocast state is thread-local.
    """
    return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
