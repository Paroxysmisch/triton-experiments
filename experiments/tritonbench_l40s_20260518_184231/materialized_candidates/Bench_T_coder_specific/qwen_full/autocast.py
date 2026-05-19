import torch
import triton
import triton.language as tl

@triton.jit
def _autocast_triton(device_type, enabled, dtype, cache_enabled, expr, kwargs):
    if isinstance(expr, str):
        return exec(expr, kwargs)
    else:
        return expr(**kwargs)

def autocast(device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            with torch.cuda.amp.autocast(device_type, enabled, dtype, cache_enabled):
                return fn(*args, **kwargs)

        return wrapper
    return decorator
