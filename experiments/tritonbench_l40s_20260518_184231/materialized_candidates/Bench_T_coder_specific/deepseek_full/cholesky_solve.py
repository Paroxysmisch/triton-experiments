import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_solve_kernel(
    B,
    L,
    n,
    b_batch_stride,
    b_row_stride,
    b_col_stride,
    l_batch_stride,
    l_row_stride,
    l_col_stride,
    upper: tl.constexpr,
    out=None,
):
    batch_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    n_stride = n * n

    l_offset = (
        batch_idx * l_batch_stride
        + col_block_idx * l_row_stride
        + 0 * l_col_stride
    )
    b_offset = (
        batch_idx * b_batch_stride
        + 0 * b_row_stride
        + col_block_idx * b_col_stride
    )
    if out is not None:
        x_offset = (
            batch_idx * b_batch_stride
            + col_block_idx * b_row_stride
            + 0 * b_col_stride
        )

    col_block_end = tl.min(col_block_idx + 1, n)

    for i in range(0, n):
        l_ptr = L + l_offset
        b_ptr = B + b_offset
        if out is not None:
            x_ptr = out + x_offset

        acc = tl.zeros((1,), dtype=tl.float32)

        for j in range(i, col_block_end):
            l_val = tl.load(l_ptr + j * l_col_stride, mask=j < n, other=0.0)
            b_val = tl.load(b_ptr + j * b_col_stride, mask=j < n, other=0.0)
            acc += l_val * b_val

        if upper:
            l_ptr += i * l_col_stride
        else:
            l_ptr += j * l_col_stride

        x_val = acc / tl.load(l_ptr)

        if out is not None:
            tl.store(x_ptr, x_val, mask=i < n)

        l_ptr += n_stride
        b_ptr += n_stride
        if out is not None:
            x_ptr += n_stride

def cholesky_solve(B, L, upper=False, *, out=None):
    if not isinstance(B, torch.Tensor):
        raise TypeError("b must be a torch.Tensor")
    if not isinstance(L, torch.Tensor):
        raise TypeError("l must be a torch.Tensor")
    if B.is_complex or L.is_complex:
        dtype = torch.complex64 if B.is_complex else torch.complex64
    else:
        dtype = torch.float32

    if B.dtype is not dtype:
        raise ValueError(f"b dtype must be {dtype}")
    if L.dtype is not dtype:
        raise ValueError(f"l dtype must be {dtype}")
    if B.shape[-2] != L.shape[-1]:
        raise ValueError("incompatible dimensions")
    if B.ndim < 2:
        raise ValueError("b must be at least 2D tensor")
    if L.ndim < 2:
        raise ValueError("l must be at least 2D tensor")
    if B.stride(-2) != 1 or B.stride(-1) != B.shape[-1]:
        B = B.contiguous()
    if L.stride(-2) != 1 or L.stride(-1) != L.shape[-1]:
        L = L.contiguous()
    if out is not None:
        if out.ndim < 2:
            raise ValueError("out must be at least 2D tensor")
        if out.shape[-2] != B.shape[-2]:
            raise ValueError("incompatible dimensions")
        if out.stride(-2) != 1 or out.stride(-1) != out.shape[-1]:
            out = out.contiguous()
        if B.dtype is not torch.complex64:
            if out.dtype is not torch.complex64:
                raise ValueError("out dtype must be complex64 if b is real")
        else:
            if out.dtype is torch.complex64:
                raise ValueError("out dtype must be real64 if b is complex")

    batch_shape = B.shape[:-2]
    n = B.shape[-1]

    if out is None:
        out = torch.empty(B.shape[:-1], dtype=B.dtype, device=B.device)

    cholesky_solve_kernel[(torch.prod(torch.tensor(batch_shape)), n)](
        B,
        L,
        n,
        B.stride(0),
        B.stride(1),
        B.stride(2),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        upper,
        out=out,
    )

    return out
