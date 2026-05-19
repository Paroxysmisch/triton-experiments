import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional
from math import prod

@triton.jit
def determinant_lu_kernel(
    a_ptr,
    n,
    p_sign_ptr,
    a_row_stride,
    a_search_start_col_idx,
    a_search_end_col_idx,
    A_BLOCK_SIZE: tl.constexpr,
    pivot: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    a_row_idx = tl.program_id(0)
    a_col_idx = tl.program_id(1)

    if a_col_idx >= a_search_start_col_idx:
        a_search_end_row_idx = min(n, a_row_idx + A_BLOCK_SIZE)

        a_row_start_ptr = a_ptr + a_row_idx * a_row_stride
        a_search_block_col_offsets = tl.arange(0, A_BLOCK_SIZE)
        a_search_block_row_offsets = tl.arange(a_col_idx, a_search_end_row_idx)
        a_search_block_ptrs = (
            a_row_start_ptr
            + a_search_block_row_offsets[:, None] * a_row_stride
            + a_search_block_col_offsets[None, :]
        )
        a_search_block = tl.load(
            a_search_block_ptrs,
            mask=(a_search_block_row_offsets[:, None] >= a_col_idx)
            & (a_search_block_col_offsets[None, :] < a_search_end_row_idx - a_col_idx),
            other=0.0,
        )

        if pivot:
            a_pivot_col_idx = tl.argmax(tl.abs(a_search_block), axis=0)
            a_pivot_val = tl.max(tl.abs(a_search_block), axis=0)
            a_pivot_col_start_ptr = a_ptr + a_col_idx
            tl.store(a_pivot_col_start_ptr, a_pivot_val)

            a_search_block /= a_pivot_val
            a_search_block_ptrs = (
                a_row_start_ptr
                + a_search_block_row_offsets[:, None] * a_row_stride
                + a_search_block_col_offsets[None, :]
            )
            tl.store(a_search_block_ptrs, a_search_block, mask=(a_search_block_row_offsets[:, None] >= a_col_idx) & (a_search_block_col_offsets[None, :] < a_search_end_row_idx - a_col_idx))

            if a_row_idx == a_col_idx:
                p_sign_start_ptr = p_sign_ptr + a_col_idx
                tl.store(p_sign_start_ptr, tl.where(a_pivot_col_idx % 2 == 0, 1.0, -1.0))

        else:
            a_row_end_idx = min(n, a_row_idx + A_BLOCK_SIZE)
            a_row_end_start_ptr = a_ptr + a_row_end_idx * a_row_stride + a_col_idx
            a_row_end = tl.load(a_row_end_start_ptr)

            a_row_start_ptr = a_ptr + a_row_idx * a_row_stride + a_col_idx
            a_row = tl.load(a_row_start_ptr)

            a_row_end_start_ptr = a_ptr + a_row_end_idx * a_row_stride + a_col_idx
            tl.store(a_row_end_start_ptr, a_row)

            a_row_start_ptr = a_ptr + a_row_idx * a_row_stride + a_col_idx
            tl.store(a_row_start_ptr, a_row_end)


def determinant_lu(
    A: Tensor,
    *,
    pivot: bool = True,
    out: Optional[Tensor] = None,
) -> Tensor:
    if not A.is_floating_point():
        raise ValueError("determinant_lu only supports floating point dtypes")

    if A.ndim < 2:
        raise ValueError("Input tensor must be at least 2D")

    dim = A.shape[-2:]
    n = max(dim)

    if dim[0] != dim[1]:
        raise ValueError("Input tensor must be a square matrix")

    if out is not None:
        if out.shape != (n,):
            raise ValueError(
                "out must be of shape (n,); expected {}, but got {}".format((n,), out.shape)
            )
        if A.dtype is torch.float32 and out.dtype is not torch.float32:
            A = A.to(torch.float32)
        if A.dtype is torch.float64 and out.dtype is not torch.float64:
            A = A.to(torch.float64)
        if A.dtype is torch.complex64 and out.dtype is not torch.complex64:
            A = A.to(torch.complex64)
        if A.dtype is torch.complex128 and out.dtype is not torch.complex128:
            A = A.to(torch.complex128)
    else:
        out = torch.empty(n, dtype=A.dtype, device=A.device)

    if pivot:
        p_sign = torch.empty(n, dtype=torch.float32, device=A.device)
    else:
        p_sign = None

    A_block_size = triton.next_power_of_2(n)
    num_warps = 4
    if A_block_size >= 2048:
        num_warps = 8
    if A_block_size >= 4096:
        num_warps = 16

    determinant_lu_kernel[(n, n)](
        A,
        n,
        p_sign,
        A.stride(0),
        0,
        n,
        num_warps=num_warps,
        A_BLOCK_SIZE=A_block_size,
        pivot=pivot,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )

    if pivot:
        out += torch.sum(torch.log(torch.abs(A.diagonal())), axis=0)
        out *= p_sign
        out = torch.exp(out)
        out = prod(out)
    else:
        out += torch.sum(torch.log(torch.abs(A.diagonal())), axis=0)
        out = torch.exp(out)
        out = prod(out)

    return out
