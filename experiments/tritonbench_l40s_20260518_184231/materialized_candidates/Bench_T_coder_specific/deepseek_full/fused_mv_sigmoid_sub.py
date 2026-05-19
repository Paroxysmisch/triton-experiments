import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import annotate

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr,  # pointer to the matrix A
    vec_ptr,  # pointer to the vector v
    other_ptr,  # pointer to the scalar b or tensor
    output_ptr,  # pointer to the output y
    M,  # number of rows in A
    N,  # number of columns in A
    K,  # number of elements in vector v
    alpha,  # scalar multiplier for b
    stride_input_row,  # how much to increase the pointer when moving by 1 row
    stride_vec,  # how much to increase the pointer when moving by 1 element in vector v
    stride_output_row,  # how much to increase the pointer when moving by 1 row in the output
    stride_other_row,  # how much to increase the pointer when moving by 1 row in the other tensor
    BLOCK_SIZE_M: tl.constexpr,  # block size in the rows of A
    BLOCK_SIZE_N: tl.constexpr,  # block size in the columns of A
    GROUP_SIZE_M: tl.constexpr,  # group size in the rows of A
):
    # Triton kernel code
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    input_ptrs = input_ptr + (
        offs_m[:, None] * stride_input_row + offs_n[None, :] * stride_vec
    )
    vec_ptrs = vec_ptr + offs_n * stride_vec

    shifter = tl.constexpr(15)
    log2e = tl.constexpr(1.4426950408889634)
    x_mask = offs_m[:, None] < M and offs_n[None, :] < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator += tl.load(
        other_ptr + (offs_m[:, None] * stride_other_row + offs_n[None, :])
    )
    if alpha != 1:
        accumulator *= alpha
    for _ in range(0, K, 4):
        vec = tl.load(vec_ptrs)
        x = tl.load(input_ptrs, mask=x_mask[None, :])
        accumulator += x * vec
        input_ptrs += K
        vec_ptrs += K

    output = accumulator
    output = (output * (1 / (1 + tl.exp(-output))))

    other_ptrs = other_ptr + offs_n * stride_other_row
    other_mask = offs_n < N
    other = tl.load(other_ptrs, mask=other_mask)
    if alpha != 1:
        other *= alpha
    output -= other

    output_mask = offs_m[:, None] < M and offs_n[None, :] < N
    tl.store(output_ptr + (offs_m[:, None] * stride_output_row + offs_n[None, :]), output, mask=output_mask)

@annotate("fused_mv_sigmoid_sub", on_device=True)
@torch.inference_mode()
def fused_mv_sigmoid_sub(
    input: Tensor,
    vec: Tensor,
    other: Tensor | float,
    alpha: float = 1,
    *,
    out: Tensor | None = None,
) -> Tensor:
    # function code
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)
    else:
        out = torch.empty_like(out)

    if isinstance(other, float):
        other = torch.tensor(
            other, dtype=input.dtype, device=input.device
        )  # scalar -> tensor
    else:
        other = other.to(dtype=input.dtype, device=input.device)

    assert input.shape[1] == vec.shape[0], "incompatible dimensions"
    assert input.is_contiguous(), "input must be contiguous"
    assert vec.is_contiguous(), "vec must be contiguous"
    assert other.is_contiguous(), "other must be contiguous"
    assert out.is_contiguous(), "out must be contiguous"

    M, N = input.shape
    K = vec.shape[0]
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fused_mv_sigmoid_sub_kernel[grid](
        input,
        vec,
        other,
        out,
        M,
        N,
        K,
        alpha,
        input.stride(0),
        vec.stride(0),
        out.stride(0),
        other.stride(0),
    )
    return out
