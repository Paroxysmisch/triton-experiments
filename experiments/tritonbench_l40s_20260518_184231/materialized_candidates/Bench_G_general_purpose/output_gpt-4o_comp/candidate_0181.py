import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_outm, stride_outn,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    rk = tl.arange(0, BLOCK_SIZE_K)

    x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk, mask=(rm[:, None] < M) & (rk[None, :] < K))
    w = tl.load(w_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk, mask=(rn[None, :] < N) & (rk[:, None] < K))

    rms_x = tl.sqrt(tl.sum(x * x, axis=1) + EPS)
    rms_w = tl.load(rms_w_ptr + rn, mask=(rn < N))

    x_normalized = x / rms_x[:, None]
    w_normalized = w / rms_w[None, :]

    acc = tl.dot(x_normalized, w_normalized)

    if RBE_EPILOGUE:
        # Apply rotary embeddings
        cos_theta = tl.cos(THETA * rk)
        sin_theta = tl.sin(THETA * rk)
        acc_rotated = acc * cos_theta + acc * sin_theta
        acc = acc_rotated

    tl.store(out_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def rms_matmul_rbe_qkv(
    q_ptr, k_ptr, v_ptr, wq_ptr, wk_ptr, wv_ptr, rms_wq_ptr, rms_wk_ptr, rms_wv_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    M, N, K,
    stride_qm, stride_qk, stride_km, stride_kn, stride_vm, stride_vk,
    stride_wqm, stride_wqn, stride_wkm, stride_wkn, stride_wvm, stride_wvn,
    stride_qoutm, stride_qoutn, stride_koutm, stride_koutn, stride_voutm, stride_voutn,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    rms_matmul_rbe(q_ptr, wq_ptr, rms_wq_ptr, q_out_ptr, M, N, K, stride_qm, stride_qk, stride_wqn, stride_wqm, stride_qoutm, stride_qoutn, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    rms_matmul_rbe(k_ptr, wk_ptr, rms_wk_ptr, k_out_ptr, M, N, K, stride_km, stride_kn, stride_wkn, stride_wkm, stride_koutm, stride_koutn, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    rms_matmul_rbe(v_ptr, wv_ptr, rms_wv_ptr, v_out_ptr, M, N, K, stride_vm, stride_vk, stride_wvn, stride_wvm, stride_voutm, stride_voutn, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)


import torch

def rms_matmul_rbe_qkv_wrapper(
    q, k, v, wq, wk, wv, rms_wq, rms_wk, rms_wv,
    start_token_position=0, use_fp8=False, rbe_epilogue=False, theta=0.0, eps=1e-5,
    block_size_m=128, block_size_n=128, block_size_k=32
):
    assert q.dtype == k.dtype == v.dtype == wq.dtype == wk.dtype == wv.dtype, "All inputs must have the same data type"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == wq.shape[-1] == wk.shape[-1] == wv.shape[-1], "All inputs must have compatible dimensions"

    M, K = q.shape
    _, N = wq.shape

    q_out = torch.empty((M, N), dtype=q.dtype, device=q.device)
    k_out = torch.empty((M, N), dtype=k.dtype, device=k.device)
    v_out = torch.empty((M, N), dtype=v.dtype, device=v.device)

    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))

    rms_matmul_rbe_qkv[grid](
        q, k, v, wq, wk, wv, rms_wq, rms_wk, rms_wv,
        q_out, k_out, v_out,
        M, N, K,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        wq.stride(0), wq.stride(1), wk.stride(0), wk.stride(1), wv.stride(0), wv.stride(1),
        q_out.stride(0), q_out.stride(1), k_out.stride(0), k_out.stride(1), v_out.stride(0), v_out.stride(1),
        start_token_position, use_fp8, rbe_epilogue, theta, eps,
        block_size_m, block_size_n, block_size_k
    )

    return q_out, k_out, v_out
