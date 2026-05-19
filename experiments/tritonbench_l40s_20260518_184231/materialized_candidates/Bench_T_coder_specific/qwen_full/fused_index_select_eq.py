import torch
import triton
import triton.language as tl

@triton.jit
def triton_fused_index_select_eq_kernel(
    X,
    I,
    Y,
    O,
    stride_x,
    stride_y,
    stride_o,
    stride_ix,
    stride_iy,
    stride_oix,
    stride_oiy,
    M,
    N,
    K,
    idx,
    IDX_MASK,
    NUM_WARPS,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    if pid_m * BLOCK_SIZE_M >= M:
        return
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    I_block_ptr = I + (offs_m[:, None] * stride_ix + offs_n[None, :] * stride_iy)
    O_block_ptr = O + (offs_m[:, None] * stride_oix + offs_n[None, :] * stride_oiy)
    if IDX_MASK:
        idx_mask = idx + offs_m * N + offs_n
        I_block = tl.load(I_block_ptr, mask=idx_mask < K, other=0)
    else:
        I_block = tl.load(I_block_ptr)
    X_block_ptr = (
        X
        + (offs_m[:, None] * stride_x + offs_n[None, :] * stride_y)
        + I_block[:, None] * K
    )
    Y_block_ptr = Y + offs_n
    X_block = tl.load(X_block_ptr)
    Y_block = tl.load(Y_block_ptr, mask=offs_n < N, other=0)
    O_block = tl.where(X_block == Y_block, 1, 0)
    tl.store(O_block_ptr, O_block, mask=idx_mask < K)


def fused_index_select_eq_triton(input, dim, index, other, *, out=None):
    dim = dim % input.ndim
    input_shape = list(input.shape)
    index_shape = list(index.shape)
    other_shape = list(other.shape) if isinstance(other, torch.Tensor) else [1]
    if len(index_shape) == 0:
        index_shape = [1]
    if len(other_shape) == 0:
        other_shape = [1]
    index_strides = [0 for _ in index_shape]
    other_strides = [0 for _ in other_shape]
    if index.ndim == 2:
        index_strides[0] = index_shape[1]
    if other.ndim == 2:
        other_strides[0] = other_shape[1]
    broadcast_shape = list(
        broadcast_shapes([input_shape[: dim + 1], index_shape, other_shape])
    )
    if broadcast_shape[dim] == 1:
        broadcast_shape[dim] = input_shape[dim]
    if broadcast_shape[-1] == 1:
        broadcast_shape[-1] = other_shape[-1]
    in_broadcast_dims = [i for i in range(len(broadcast_shape)) if i != dim]
    out_shape = list(input.shape)
    out_shape[in_broadcast_dims] = broadcast_shape[in_broadcast_dims]
    out = torch.empty(out_shape, dtype=torch.bool, device=input.device)
    if out.numel() == 0:
        return out
    index_shape = list(index.shape)
    if len(index_shape) == 0:
        index_shape = [1]
    N = index_shape[-1]
    M = triton.cdiv(input.size(dim), 1)
    K = index.size(dim)
    if IDX_MASK:
        idx = torch.arange(0, N, device=index.device).expand(M, N)
    else:
        idx = None
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    triton_fused_index_select_eq_kernel[grid](
        input,
        index,
        other,
        out,
        input.stride(dim),
        other.stride(0),
        out.stride(0),
        index.stride(0),
        index.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
        K,
        idx,
        IDX_MASK,
        NUM_WARPS=NUM_WARPS,
    )
    return out
