import torch
import triton
import triton.language as tl


@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    **meta
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N

    # Create pointers for output
    out_ptrs = out_ptr + (m_offset * stride_om + n_offset * stride_on)

    # Accumulate partial results in registers
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # For row-wise RMS normalization, partial sums to compute mean of squares
    # Each row in the tile accumulates sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    # Loop over K in steps of BLOCK_SIZE_K
    for k_block in range(0, K, BLOCK_SIZE_K):
        # Pointers for x and w
        x_ptrs = x_ptr + (m_offset * stride_xm + (k_block) * stride_xk)
        w_ptrs = w_ptr + ((k_block) * stride_wk + n_offset * stride_wn)

        # Load x and w
        x_tile = tl.load(
            x_ptrs + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_xm)
            + tl.arange(0, BLOCK_SIZE_K)[None, :]
       , mask=(m_offset + tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
              (k_block + tl.arange(0, BLOCK_SIZE_K) < K)[None, :],
        other=0.0)

        w_tile = tl.load(
            w_ptrs + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_wk)
            + tl.arange(0, BLOCK_SIZE_N)[None, :]
        , mask=(k_block + tl.arange(0, BLOCK_SIZE_K) < K)[:, None] &
               (n_offset + tl.arange(0, BLOCK_SIZE_N) < N)[None, :],
        other=0.0)

        # Accumulate sum of squares for RMS
        sum_sq += tl.sum(x_tile * x_tile, 1)

        # Multiply partial product
        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    # Final RMS for each row
    denom = tl.sqrt(sum_sq / K + EPS)
    # Multiply by RMS of weight if applicable (broadcasted scalar)
    w_rms = tl.load(rms_w_ptr, mask=[True], other=1.0)
    scale = w_rms / denom

    # Scale the accumulation
    # Expand scale to shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
    scale_matrix = scale[:, None]
    acc = acc * scale_matrix

    # Optional rotary embedding epilogue
    if RBE_EPILOGUE != 0:
        # Each row has a global token index to compute rotation
        global_m = m_offset + tl.arange(0, BLOCK_SIZE_M)
        cos_theta = tl.cos((global_m + start_token_position) * THETA)
        sin_theta = tl.sin((global_m + start_token_position) * THETA)
        valid_m = global_m < M
        valid_n = (n_offset + tl.arange(0, BLOCK_SIZE_N)) < N

        # Typically we apply RBE on half dimension, but here we apply
        # a simple version for demonstration
        for i in range(BLOCK_SIZE_M):
            if valid_m[i]:
                for j in range(BLOCK_SIZE_N):
                    if valid_n[j]:
                        old_val = acc[i, j]
                        acc[i, j] = old_val * cos_theta[i] - old_val * sin_theta[i]  # toy example

    # Store results
    mask_m = (m_offset + tl.arange(0, BLOCK_SIZE_M)) < M
    mask_n = (n_offset + tl.arange(0, BLOCK_SIZE_N)) < N
    tl.store(
        out_ptrs + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_om)
        + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_on,
        acc.to(tl.float16) if not USE_FP8 else acc,  # FP8 or FP16
        mask=mask_m[:, None] & mask_n[None, :]
    )


@triton.jit
def rms_matmul_rbe_qkv(
    x_ptr, wq_ptr, wk_ptr, wv_ptr,
    rms_wq_ptr, rms_wk_ptr, rms_wv_ptr,
    out_q_ptr, out_k_ptr, out_v_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wqk, stride_wqn,
    stride_wkk, stride_wkn,
    stride_wvk, stride_wvn,
    stride_oqm, stride_oqn,
    stride_okm, stride_okn,
    stride_ovm, stride_ovn,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    **meta
):
    # Q
    rms_matmul_rbe[grid(meta['grid'])](
        x_ptr, wq_ptr, rms_wq_ptr, out_q_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wqk, stride_wqn,
        stride_oqm, stride_oqn,
        start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    # K
    rms_matmul_rbe[grid(meta['grid'])](
        x_ptr, wk_ptr, rms_wk_ptr, out_k_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wkk, stride_wkn,
        stride_okm, stride_okn,
        start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    # V
    rms_matmul_rbe[grid(meta['grid'])](
        x_ptr, wv_ptr, rms_wv_ptr, out_v_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wvk, stride_wvn,
        stride_ovm, stride_ovn,
        start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )


def rms_matmul_rbe_qkv_wrapper(
    x, wq, wk, wv,
    rms_wq, rms_wk, rms_wv,
    start_token_position=0,
    use_fp8=False,
    apply_rbe=False,
    theta=0.001,
    eps=1e-6,
    block_size_m=64,
    block_size_n=64,
    block_size_k=32
):
    assert x.dtype in [torch.float16, torch.float32], "Only FP16/FP32 for x"
    assert wq.dtype in [torch.float16, torch.float32], "Only FP16/FP32 for wq"
    M, K = x.shape
    Kwq, Nq = wq.shape
    Kwk, Nk = wk.shape
    Kwv, Nv = wv.shape
    assert K == Kwq == Kwk == Kwv, "K dimension mismatch"
    assert Nq == Nk == Nv, "N dimension mismatch"

    out_q = torch.empty((M, Nq), dtype=x.dtype, device=x.device)
    out_k = torch.empty((M, Nk), dtype=x.dtype, device=x.device)
    out_v = torch.empty((M, Nv), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(Nq, meta['BLOCK_SIZE_N']))

    rms_matmul_rbe_qkv[grid({'BLOCK_SIZE_M': block_size_m,
                             'BLOCK_SIZE_N': block_size_n,
                             'BLOCK_SIZE_K': block_size_k,
                             'grid': grid})](
        x, wq, wk, wv,
        rms_wq, rms_wk, rms_wv,
        out_q, out_k, out_v,
        M, Nq, K,
        x.stride(0), x.stride(1),
        wq.stride(0), wq.stride(1),
        wk.stride(0), wk.stride(1),
        wv.stride(0), wv.stride(1),
        out_q.stride(0), out_q.stride(1),
        out_k.stride(0), out_k.stride(1),
        out_v.stride(0), out_v.stride(1),
        start_token_position,
        int(use_fp8),
        int(apply_rbe),
        theta,
        eps,
        block_size_m,
        block_size_n,
        block_size_k
    )
    return out_q, out_k, out_v
