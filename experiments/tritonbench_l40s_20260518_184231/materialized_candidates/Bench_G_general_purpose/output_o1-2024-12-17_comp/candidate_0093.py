import torch
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16(
    X_ptr, 
    Y_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x_int8 = tl.load(X_ptr + offs, mask=mask, other=0)
    # Interpret the input as float8. Triton does not have float8, so this is conceptual.
    # Here, we simply cast to float16 for demonstration.
    x_f16 = x_int8.to(tl.float16)
    tl.store(Y_ptr + offs, x_f16, mask=mask)

def f8_to_f16(X: torch.Tensor) -> torch.Tensor:
    assert X.dtype == torch.int8, "Input tensor must be of type torch.int8"
    assert X.is_cuda, "Input tensor must be on a CUDA device"
    out = torch.empty(X.shape, dtype=torch.float16, device=X.device)
    N = X.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel_f8_to_f16[grid](X, out, N, BLOCK_SIZE)
    return out

@triton.jit
def kernel_f16_to_f8(
    X_ptr, 
    Y_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x_val = tl.load(X_ptr + offs, mask=mask, other=0)
    # Conceptual float8 representation, cast to int8 for storage
    y_int8 = x_val.to(tl.int8)
    tl.store(Y_ptr + offs, y_int8, mask=mask)

def f16_to_f8(X: torch.Tensor) -> torch.Tensor:
    assert X.dtype in [torch.float16, torch.float32], "Input must be float16 or float32"
    assert X.is_cuda, "Input tensor must be on a CUDA device"
    out = torch.empty(X.shape, dtype=torch.int8, device=X.device)
    N = X.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    kernel_f16_to_f8[grid](X, out, N, BLOCK_SIZE)
    return out
