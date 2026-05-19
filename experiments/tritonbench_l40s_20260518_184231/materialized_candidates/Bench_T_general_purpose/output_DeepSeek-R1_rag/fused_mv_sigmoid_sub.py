import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, output_ptr,
    n_rows, vec_size,
    alpha,
    input_row_stride, input_col_stride,
    vec_stride,
    other_stride,
    output_stride,
    BLOCK_SIZE_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    acc = tl.zeros((1,), dtype=tl.float32)
    for col_block in range(0, vec_size, BLOCK_SIZE_M):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE_M)
        mask = col_offsets < vec_size

        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets * input_col_stride
        vec_ptrs = vec_ptr + col_offsets * vec_stride

        a = tl.load(input_ptrs, mask=mask, other=0.0)
        b = tl.load(vec_ptrs, mask=mask, other=0.0)
        acc += tl.sum(a * b)

    s = 1.0 / (1.0 + tl.exp(-acc))
    other_val = tl.load(other_ptr + row_idx * other_stride)
    result = s - alpha * other_val

    output_ptr_row = output_ptr + row_idx * output_stride
    tl.store(output_ptr_row, result)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2, "input must be 2D"
    assert vec.dim() == 1, "vec must be 1D"
    n, m = input.shape
    assert vec.size(0) == m, "input and vec have incompatible shapes"

    if isinstance(other, torch.Tensor):
        if other.dim() != 0:
            other = other.broadcast_to((n,))
        other = other.contiguous()
    else:
        other = torch.full((n,), other, dtype=input.dtype, device=input.device)

    if out is None:
        out = torch.empty((n,), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n,), "out has incorrect shape"
        assert out.dtype == input.dtype, "out dtype mismatch"
        assert out.device == input.device, "out device mismatch"

    BLOCK_SIZE_M = 1024
    grid = (n,)
    fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m,
        alpha,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0),
        out.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return out
