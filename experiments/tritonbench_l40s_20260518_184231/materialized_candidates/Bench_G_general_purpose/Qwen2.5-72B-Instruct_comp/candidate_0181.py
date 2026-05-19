import triton
import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_rmsw,
    stride_om, stride_on,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    rms_w_ptrs = rms_w_ptr + (offs_k * stride_rmsw)

    x = tl.load(x_ptrs)
    w = tl.load(w_ptrs)
    rms_w = tl.load(rms_w_ptrs)

    # Compute RMS normalization
    rms_x = tl.sqrt(tl.sum(x * x, axis=1) / K + EPS)
    x_normalized = x / rms_x[:, None]

    # Matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x_block = x_normalized[:, k:k + BLOCK_SIZE_K]
        w_block = w[k:k + BLOCK_SIZE_K, :]
        acc += tl.dot(x_block, w_block)

    # Apply RMS weight
    acc *= rms_w

    # Apply rotary embeddings if RBE_EPILOGUE is True
    if RBE_EPILOGUE:
        theta = THETA * (offs_bn + start_token_position)
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)
        acc_rotated = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_rotated[:, :BLOCK_SIZE_N // 2] = acc[:, :BLOCK_SIZE_N // 2] * cos_theta - acc[:, BLOCK_SIZE_N // 2:] * sin_theta
        acc_rotated[:, BLOCK_SIZE_N // 2:] = acc[:, :BLOCK_SIZE_N // 2] * sin_theta + acc[:, BLOCK_SIZE_N // 2:] * cos_theta
        acc = acc_rotated

    # Store the result
    out_ptrs = out_ptr + (offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on)
    tl.store(out_ptrs, acc)

@triton.jit
def rms_matmul_rbe_qkv(
    x_ptr, w_q_ptr, w_k_ptr, w_v_ptr, rms_w_ptr, out_q_ptr, out_k_ptr, out_v_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wq_n, stride_wq_k,
    stride_wk_n, stride_wk_k,
    stride_wv_n, stride_wv_k,
    stride_rmsw,
    stride_oq_m, stride_oq_n,
    stride_ok_m, stride_ok_n,
    stride_ov_m, stride_ov_n,
    start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    rms_matmul_rbe(x_ptr, w_q_ptr, rms_w_ptr, out_q_ptr, M, N, K, stride_xm, stride_xk, stride_wq_n, stride_wq_k, stride_rmsw, stride_oq_m, stride_oq_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    rms_matmul_rbe(x_ptr, w_k_ptr, rms_w_ptr, out_k_ptr, M, N, K, stride_xm, stride_xk, stride_wk_n, stride_wk_k, stride_rmsw, stride_ok_m, stride_ok_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    rms_matmul_rbe(x_ptr, w_v_ptr, rms_w_ptr, out_v_ptr, M, N, K, stride_xm, stride_xk, stride_wv_n, stride_wv_k, stride_rmsw, stride_ov_m, stride_ov_n, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

import torch

def rms_matmul_rbe_qkv_wrapper(x, w_q, w_k, w_v, rms_w, start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    # Ensure input tensors are on the same device and have the correct data type
    assert x.device == w_q.device == w_k.device == w_v.device == rms_w.device, "All tensors must be on the same device"
    assert x.dtype == w_q.dtype == w_k.dtype == w_v.dtype == rms_w.dtype, "All tensors must have the same data type"

    # Ensure input tensors have the correct shape
    M, K = x.shape
    _, N = w_q.shape
    assert w_k.shape == (K, N) and w_v.shape == (K, N), "Weight matrices must have shape (K, N)"
    assert rms_w.shape == (K,), "RMS weight vector must have shape (K,)"

    # Initialize output tensors
    out_q = torch.empty((M, N), device=x.device, dtype=x.dtype)
    out_k = torch.empty((M, N), device=x.device, dtype=x.dtype)
    out_v = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Set up grid configuration
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch the kernel
    rms_matmul_rbe_qkv[grid](
        x, w_q, w_k, w_v, rms_w, out_q, out_k, out_v,
        M, N, K,
        x.stride(0), x.stride(1),
        w_q.stride(1), w_q.stride(0),
        w_k.stride(1), w_k.stride(0),
        w_v.stride(1), w_v.stride(0),
        rms_w.stride(0),
        out_q.stride(0), out_q.stride(1),
        out_k.stride(0), out_k.stride(1),
        out_v.stride(0), out_v.stride(1),
        start_token_position, USE_FP8, RBE_EPILOGUE, THETA, EPS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return out_q, out_k, out_v
