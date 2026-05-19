import argparse
import math
from typing import Optional
import torch
from torch.utils.benchmark import Timer
import triton
import triton.language as tl
from triton.testing import ALL_DTYPES, no_grad_func, run
from triton.tools import make_sure_has_torch_args
from tqdm import trange

from .utils import assert_


@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to):
    phi = tl.array([-0.5, -0.5, 0.5, 0.5], dtype=tl.float32) * (to - from_)
    bloat = tl.array([to, from_, to, from_], dtype=tl.float32)
    offsets = philox_offset + tl.arange(0, BLOCK)
    v0, v1, v2, v3 = tl.rand(philox_seed, offsets)
    v = v0 + v1 + v2 + v3 + phi
    mask = (offsets < N)
    tl.store(out_ptr + offsets, (v + bloat) * mask, mask=mask)


@make_sure_has_torch_args
def uniform_(
        out: torch.Tensor,
        N: int,
        from_=0.0,
        to=1.0,
        generator=None,
        device=None,
        non_blocking=False,
        *,
        phi=0,
        out_=None,
        requires_grad=False,
        pin_memory=False,
):
    if device is None:
        device = torch._C.get_device_index()
    _, philox_seed, philox_offset, _ = torch.ops.torch.philox_cuda_seed_offset(
        N.item(), device, phi
    )

    if generator is not None:
        raise NotImplementedError('Generator support not implemented.')

    num_warps = max(1, N // BLOCK // 2)
    grid = (num_warps, )
    uniform_kernel[grid](
        out,
        N,
        philox_seed,
        philox_offset,
        torch.tensor(from_, device=out.device, dtype=out.dtype),
        torch.tensor(to, device=out.device, dtype=out.dtype),
    )
