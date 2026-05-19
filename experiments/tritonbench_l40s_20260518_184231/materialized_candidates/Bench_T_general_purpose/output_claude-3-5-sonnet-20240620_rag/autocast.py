import torch
import triton
import triton.language as tl
import contextlib
import threading

@triton.jit
def autocast_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dtype_code: tl.constexpr,  # 0: float32, 1: float16, 2: bfloat16
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert to specified dtype
    if dtype_code == 1:  # float16
        x = tl.float16(x)
    elif dtype_code == 2:  # bfloat16
        x = tl.bfloat16(x)
        
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

class AutocastTriton(contextlib.ContextDecorator):
    _thread_local = threading.local()
    
    def __init__(self, device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
        self.device_type = device_type
        self.enabled = enabled
        self.dtype = dtype or torch.float16  # default to float16 if not specified
        self.cache_enabled = cache_enabled
        
    def __enter__(self):
        if not hasattr(AutocastTriton._thread_local, 'stack'):
            AutocastTriton._thread_local.stack = []
        
        if self.enabled and self.device_type == "cuda":
            AutocastTriton._thread_local.stack.append(self)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.device_type == "cuda":
            AutocastTriton._thread_local.stack.pop()
        
    def __call__(self, func):
        def wrapped(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapped
    
    def convert_tensor(self, tensor):
        if not self.enabled or not tensor.is_cuda:
            return tensor
            
        output = torch.empty_like(tensor)
        n_elements = tensor.numel()
        
        # Determine dtype code
        dtype_code = 0  # default float32
        if self.dtype == torch.float16:
            dtype_code = 1
        elif self.dtype == torch.bfloat16:
            dtype_code = 2
            
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        autocast_kernel[grid](
            tensor, 
            output,
            n_elements,
            dtype_code,
            BLOCK_SIZE=1024
        )
        
        return output

def autocast(device_type="cuda", enabled=True, dtype=None, cache_enabled=True):
    """
    Triton implementation of autocast context manager/decorator.
    
    Args:
        device_type (str): Device type, currently only "cuda" is supported
        enabled (bool): Whether autocast is enabled
        dtype (torch.dtype): Target dtype for autocasting (default: torch.float16)
        cache_enabled (bool): Whether to enable caching of autocast operations
        
    Returns:
        ContextManager: Context manager/decorator for mixed precision operations
    """
    return AutocastTriton(device_type, enabled, dtype, cache_enabled)
