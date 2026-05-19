import torch
import triton
import triton.language as tl
from typing import Optional

# Triton kernel that performs matrix multiplication with RMS normalization and optional rotary embeddings.
@triton.jit
def rms_matmul_rbe(
    x_ptr,
    w_ptr,
    rms_w_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_x_k: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    start_token_position: tl.constexpr,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # The epilogue is only present in some of the calls to this kernel. We need to know this at compile time.
    # If the epilogue is present, it uses rotary embeddings.
    # If the epilogue is absent, it does not use rotary embeddings.
    pid = tl.program_id(axis=0)
    if start_token_position + pid * BLOCK_SIZE_M >= M:
        return
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_base_ptrs = x_ptr + (offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_base_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    rms_w_ptrs = rms_w_ptr + offs_k
    x_ptrs = x_base_ptrs + tl.arange(0, BLOCK_SIZE_K)
    w_ptrs = w_base_ptrs + tl.arange(0, BLOCK_SIZE_K)
    accumulator_dtype = tl.float32 if USE_FP8 else tl.float16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float8e5, bitcast=True)
        x = (x * rms_w).to(accumulator_dtype)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if RBE_EPILOGUE:
        # This is the case for QK and V branches. Apply RoPE here.
        theta = (offs_n[None, :] + THETA) * (THETA + 0.5)
        rms_w = tl.load(rms_w_ptr + offs_n, mask=offs_n < N, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
        rms_w = rms_w.to(accumulator_dtype)
        out = accumulator * rms_w
        out = out / theta
        out = out.to(accumulator_dtype)
        out = tl.where(out_mask, out, 0.0)
        out = out.to(tl.float16)
        tl.store(out_ptrs, out, mask=out_mask)
    else:
        # This is the case for the inner product of Q and K^T, which is present in the attention branch.
        # Do not apply RoPE here.
        accumulator = accumulator.to(tl.float32)
        out = accumulator / (M * EPS)
        out = tl.where(out_mask, out, 0.0)
        tl.store(out_ptrs, out, mask=out_mask)

# Triton kernel that performs three separate matrix multiplications for Q, K, and V matrices.
@triton.jit
def rms_matmul_rbe_qkv(
    q_ptr,
    k_ptr,
    v_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    rms_w_q_ptr,
    rms_w_k_ptr,
    rms_w_v_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_m: tl.constexpr,
    stride_q_k: tl.constexpr,
    stride_w_q_k: tl.constexpr,
    stride_w_q_n: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_m: tl.constexpr,
    stride_k_k: tl.constexpr,
    stride_w_k_k: tl.constexpr,
    stride_w_k_n: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_m: tl.constexpr,
    stride_v_k: tl.constexpr,
    stride_w_v_k: tl.constexpr,
    stride_w_v_n: tl.constexpr,
    stride_out_q_b: tl.constexpr,
    stride_out_q_m: tl.constexpr,
    stride_out_q_n: tl.constexpr,
    stride_out_k_b: tl.constexpr,
    stride_out_k_m: tl.constexpr,
    stride_out_k_n: tl.constexpr,
    stride_out_v_b: tl.constexpr,
    stride_out_v_m: tl.constexpr,
    stride_out_v_n: tl.constexpr,
    start_token_position: tl.constexpr,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE_Q: tl.constexpr,
    RBE_EPILOGUE_K: tl.constexpr,
    RBE_EPILOGUE_V: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # This kernel calls the rms_matmul_rbe kernel three times, each time with a different set of inputs.
    # This way we only pay the cost of setting up the kernel launch overhead three times, rather than setting it up once per matrix multiply.
    rms_matmul_rbe(
        q_ptr,
        w_q_ptr,
        rms_w_q_ptr,
        out_q_ptr,
        M,
        N,
        K,
        stride_q_b,
        stride_q_m,
        stride_q_k,
        stride_w_q_k,
        stride_w_q_n,
        stride_out_q_b,
        stride_out_q_m,
        stride_out_q_n,
        start_token_position,
        USE_FP8,
        RBE_EPILOGUE_Q,
        THETA,
        EPS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    rms_matmul_rbe(
        k_ptr,
        w_k_ptr,
        rms_w_k_ptr,
        out_k_ptr,
        M,
        N,
        K,
        stride_k_b,
        stride_k_m,
        stride_k_k,
        stride_w_k_k,
        stride_w_k_n,
        stride_out_k_b,
        stride_out_k_m,
        stride_out_k_n,
        start_token_position,
        USE_FP8,
        RBE_EPILOGUE_K,
        THETA,
        EPS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    rms_matmul_rbe(
        v_ptr,
        w_v_ptr,
        rms_w_v_ptr,
        out_v_ptr,
        M,
        N,
        K,
        stride_v_b,
        stride_v_m,
        stride_v_k,
        stride_w_v_k,
        stride_w_v_n,
        stride_out_v_b,
        stride_out_v_m,
        stride_out_v_n,
        start_token_position,
        USE_FP8,
        RBE_EPILOGUE_V,
        THETA,
        EPS,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

# Wrapper function that sets up the input and output tensors and calls the rms_matmul_rbe_qkv kernel.
def rms_matmul_rbe_qkv_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    rms_w_q: torch.Tensor,
    rms_w_k: torch.Tensor,
    rms_w_v: torch.Tensor,
    use_fp8: bool,
    theta: int,
    eps: float,
    device: Optional[torch.device] = None,
):
    # This function sets up the input and output tensors
