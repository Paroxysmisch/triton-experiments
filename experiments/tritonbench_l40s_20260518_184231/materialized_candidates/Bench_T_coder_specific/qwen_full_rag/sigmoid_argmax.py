import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._C import _assert
from typing import Optional


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-x))


@triton.jit
def sigmoid_argmax_kernel(input, idx, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset_n = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) % N
    offset_k = tl.arange(0, BLOCK_SIZE) % K
    offset = offset_n[:, None] * K + offset_k[None, :]
    mask_n = offset_n[:, None] < N
    mask_k = offset_k[None, :] < K
    mask = mask_n & mask_k
    input_val = tl.load(input + offset, mask=mask).to(tl.float32)
    sigmoid_val = sigmoid(input_val)
    argmax = tl.argmax(sigmoid_val, axis=1)
    tl.store(idx + offset_n, argmax)


def sigmoid_argmax(input: Tensor, dim=None, keepdim=False) -> Tensor:
    _assert(
        dim is None or dim >= -input.ndim and dim < input.ndim,
        "Invalid dim"
    )
    if dim is None:
        input = input.flatten()
        dim = 0
    else:
        _assert(
            input.size(dim) != 0,
            "Cannot perform reduction on a zero-size dimension"
        )

    ndim = input.ndim
    shape = list(input.shape)
    idx_shape = list(input.shape)
    idx_shape[dim] = 1

    N = 1
    K = input.size(dim)
    for i in range(dim):
        N *= shape[i]
        shape[i] = 1
    for i in range(dim + 1, ndim):
        N *= shape[i]
        shape[i] = 1

    output = torch.ones(N, device=input.device, dtype=torch.int64)
    input = input.contiguous()

    idx = torch.zeros(idx_shape, device=input.device, dtype=torch.int64)
    if K <= 1024:
        BLOCK_SIZE = K
        num_warps = 4
    else:
        BLOCK_SIZE = 1024
        num_warps = 8

    grid = (triton.cdiv(N, BLOCK_SIZE), )

    sigmoid_argmax_kernel[grid](
        input, idx,
        N, K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if not keepdim:
        output = torch.squeeze(output, dim)
    return output.to(torch.int64)
