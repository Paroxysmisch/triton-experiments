import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def rms_matmul_rbe(
    x_ptr,
    w_ptr,
    rms_w_ptr,
    y_ptr,
    stride_x_batch,
    stride_x_head,
    stride_x_m,
    stride_w_head,
    stride_w_n,
    stride_w_k,
    stride_rms_head,
    stride_y_batch,
    stride_y_head,
    stride_y_m,
    stride_y_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROTARY_EMBEDDINGS_BY_HEAD_DIM: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m_block = tl.program_id(axis=2)
    pid_n_block = tl.program_id(axis=3)

    m_offset = pid_m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_block_ptr = (
        x_ptr
        + pid_batch * stride_x_batch
        + pid_head * stride_x_head
        + m_offset[:, None] * stride_x_m
    )
    w_block_ptr = (
        w_ptr
        + pid_head * stride_w_head
        + n_offset[None, :] * stride_w_n
        + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_w_k
    )
    rms_w_ptr = rms_w_ptr + pid_head * stride_rms_head
    y_block_ptr = (
        y_ptr
        + pid_batch * stride_y_batch
        + pid_head * stride_y_head
        + m_offset[:, None] * stride_y_m
        + n_offset[None, :] * stride_y_n
    )

    rms_w = tl.load(rms_w_ptr) ** -0.5

    m_sums = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, tl.cdiv(HEAD_SIZE, BLOCK_SIZE_K)):
        x = tl.load(
            x_block_ptr,
            mask=(m_offset[:, None] < M) & (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :] < HEAD_SIZE),
            other=0.0,
        ).to(tl.float32)
        w = tl.load(w_block_ptr).to(tl.float32)
        m_sums += x * w
        x_block_ptr += BLOCK_SIZE_K * stride_w_k
        w_block_ptr += BLOCK_SIZE_K * stride_w_k

    m = tl.sum(m_sums * rms_w[None, :], axis=1)

    if MUL_ROTARY_EMBEDDINGS_BY_HEAD_DIM:
        m *= HEAD_SIZE ** 0.5

    tl.store(
        y_block_ptr,
        m[None, :],
        mask=(n_offset[None, :] < N),
    )


def rms_matmul_rbe_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    rms_w: torch.Tensor,
    mul_rotary_embeddings_by_head_dim: bool,
    theta: float,
) -> torch.Tensor:
    batch_size, n_heads, M, HEAD_SIZE = x.shape
    N, HEAD_SIZE = w.shape

    assert x.shape[1] == w.shape[1]  # batch size must be the same
    assert (
        x.is_contiguous() and w.is_contiguous()
    )  # both tensors must be contiguous

    y = torch.empty(
        (batch_size, n_heads, M, N),
        device=w.device,
        dtype=torch.float32 if w.dtype == torch.float32 else torch.float16,
    )

    if w.dtype == torch.float32:
        rms_w = rms_w.float()

    if version.parse(triton.__version__) >= version.parse("2.1.0"):
        grid = lambda META: (
            batch_size,
            n_heads,
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
    else:
        grid = (
            batch_size,
            n_heads,
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    rms_matmul_rbe[grid](
        x,
        w,
        rms_w,
        y,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        rms_w.stride(0),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        MUL_ROTARY_EMBEDDINGS_BY_HEAD_DIM=mul_rotary_embeddings_by_head_dim,
        THETA=theta,
    )

    return y.to(dtype=x.dtype)
