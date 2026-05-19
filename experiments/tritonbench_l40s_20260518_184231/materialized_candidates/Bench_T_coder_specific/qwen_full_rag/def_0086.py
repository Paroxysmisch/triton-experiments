import logging
import torch
import triton
import triton.language as tl
from triton.language.math import log, tanh


@triton.jit
def log_tanh_forward(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """fwd compute"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = log(x) 
    y = tanh(y)
    tl.store(y_ptr + offsets, y, mask=mask)


def log_tanh(input, out=None) -> torch.Tensor:
    logging.debug("GEMS LOG_TANH")
    assert all((i > 0 for i in input.flatten())), "All input elements must be positive"
    if out == None:
        return log_tanh_func(input)
    else:
        log_tanh_func(input, out=out)
        return out


spec = {0: (1024, "BLOCK_SIZE")}


class LogTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, out=None):
        logging.debug("GEMS LOG_TANH FORWARD")
        assert A.is_contiguous(), "Input must be contiguous"
        M = A.numel()
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE"]), )
        if out == None:
            out = torch.empty_like(A)
        else:
            assert out.is_contiguous(), "Output must be contiguous"
        log_tanh_forward[grid](A, out, M, **spec)
        ctx.save_for_backward(out)
        return out


def log_tanh_func(A, out=None):
    return LogTanh.apply(A, out)
