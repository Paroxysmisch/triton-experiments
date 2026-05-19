import torch
import triton
import triton.language as tl

@triton.jit
def _hstack_div_kernel(Z, Y, X, M, N, C, divisor, rounding_mode, **meta):
    c = tl.program_id(0)
    if c < C:
        y = tl.arange(0, M)
        x = tl.arange(0, N)
        mask = y[:, None] * N + x[None, :] < M * N
        div = tl.load(divisor + c).to(tl.float32)
        if rounding_mode == "trunc":
            result = tl.sum((mask * (X + y[:, None] * N + x[None, :]).to(tl.float32) / div), 1)
        elif rounding_mode == "floor":
            result = tl.sum((mask * (X + y[:, None] * N + x[None, :]).to(tl.float32) / div), 1) + 0.5
        elif rounding_mode is None:
            result = tl.sum((mask * (X + y[:, None] * N + x[None, :]).to(tl.float32) / div), 1)
        tl.store(Y + c * M + y, result)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    if len(tensors) == 0:
        raise RuntimeError("torch.hstack_div: expected a non-empty TensorList")

    if out is None:
        out = torch.empty_like(tensors[0], dtype=tensors[0].dtype, device=tensors[0].device)
    else:
        assert out.shape == tensors[0].shape and out.device == tensors[0].device

    if len(tensors) == 1:
        if isinstance(divisor, torch.Tensor):
            assert divisor.ndim == 0, "when len(tensors) == 1, divisor must be a number"
            return torch.tensor(tensors[0] / divisor, device=tensors[0].device, dtype=tensors[0].dtype)
        else:
            return torch.tensor(tensors[0] / divisor, device=tensors[0].device, dtype=tensors[0].dtype)

    M = out.numel() // out.shape[-1]
    N = out.shape[-1]
    C = len(tensors)

    grid = (C,)
    with torch.cuda.device(tensors[0].device):
        _hstack_div_kernel[grid](M, N, C, divisor, rounding_mode)

    return out
