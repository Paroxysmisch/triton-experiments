import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 32},
        num_stages=5,
        num_warps=2,
    )],
    key=["n_ctx_q", "n_ctx_k", "d_model"],
)
@triton.jit
def _score_kernel(
    q_ptr, k_ptr, mask_ptr, out_ptr,
    n_ctx_q, n_ctx_k, d_model,
    stride_ctx_q, stride_ctx_k, stride_d,
    stride_out_q, stride_out_k,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)

    grid_n = (n_ctx_k + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = tl.max_contiguous(tl.multiple_of(rm % n_ctx_q, BLOCK_M), BLOCK_M)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn = tl.max_contiguous(tl.multiple_of(rn % n_ctx_k, BLOCK_N), BLOCK_N)

    acc_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    rd = tl.arange(0, BLOCK_D)

    q_ptr_tile = q_ptr + (rm[:, None] * stride_ctx_q + rd[None, :] * stride_d)
    k_ptr_tile = k_ptr + (rd[:, None] * stride_d + rn[None, :] * stride_ctx_k)

    for d_max_offset in range(d_model, 0, -BLOCK_D):
        q_tile = tl.load(q_ptr_tile, mask=rd[None, :] < d_max_offset, other=0.0)
        k_tile = tl.load(k_ptr_tile, mask=rd[:, None] < d_max_offset, other=0.0)

        acc_tile += tl.dot(q_tile, k_tile)

        q_ptr_tile += BLOCK_D * stride_d
        k_ptr_tile += BLOCK_D * stride_d

    acc_tile *= sm_scale

    mask = (rm < n_ctx_q)[:, None] & (rn < n_ctx_k)[None, :]
    mask_tile = tl.load(mask_ptr + rm[:, None] * stride_out_q + rn[None, :] * stride_out_k, mask=mask, other=0.0)

    acc_tile = acc_tile * mask_tile

    out_offset_tile = rm[:, None] * stride_out_q + rn[None, :] * stride_out_k
    out_ptr_tile = out_ptr + out_offset_tile

    tl.store(out_ptr_tile, acc_tile, mask=mask)

def get_score(query, key, mask, sm_scale):
    device = query.device

    if query.stride(0) > 1 and query.stride(1) > 1:
        query = query.contiguous()
    if key.stride(0) > 1 and key.stride(1) > 1:
        key = key.contiguous()

    n_ctx_q, d_model = query.shape
    n_ctx_k, d_model_k = key.shape
    assert d_model == d_model_k, f"{query.shape=} {key.shape=}"

    out = torch.empty((n_ctx_q, n_ctx_k), device=device, dtype=query.dtype)

    stride_d = query.stride(1)
    assert stride_d == key.stride(1), f"{stride_d=}, {key.stride(1)=}"

    def grid(META):
        return (
            triton.cdiv(n_ctx_q, META["BLOCK_M"])
            * triton.cdiv(n_ctx_k, META["BLOCK_N"]),
        )

    try:
        _score_kernel[grid](
            query,
            key,
            mask,
            out,
            n_ctx_q,
            n_ctx_k,
            d_model,
            query.stride(0),
            key.stride(0),
            stride_d,
            out.stride(0),
            out.stride(1),
            sm_scale
        )
    except triton.OutOfResources:
        # Reduce block sizes and retry if out of resources
        _score_kernel[grid](
            query,
            key,
            mask,
            out,
            n_ctx_q,
            n_ctx_k,
            d_model,
            query.stride(0),
            key.stride(0),
            stride_d,
            out.stride(0),
            out.stride(1),
            sm_scale,
            BLOCK_M=32, BLOCK_N=16, BLOCK_D=16
        )

    return out
