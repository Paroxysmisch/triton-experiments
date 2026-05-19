import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autotune import measure
from torch.distributed.rpc import RRef
from torch.distributed.rpc import rpc_sync
from torch.distributed.rpc import RpcTask

@triton.jit
def rms_matmul_rbe(
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_x_batch, stride_x_M, stride_x_K,
    stride_w_K, stride_w_N,
    stride_rms_w_N,
    stride_out_batch, stride_out_N, stride_out_K,
    start_token_position: tl.constexpr,
    USE_FP8: tl.constexpr,
    RBE_EPILOGUE: tl.constexpr,
    THETA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute the matrix expression: out = (rms(x) * rms_w) @ w.
    x shape: (batch, M, K)
    w shape: (K, N)
    rms_w shape: (N,)
    out shape: (batch, M, N)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    # compute rm
    rm = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        x_mask = k_offset < K
        x_ptr_mask = k_offset * stride_x_K
        x_2d = tl.load(x_ptr + x_ptr_mask, mask=x_mask, other=0)
        x_rm = tl.sqrt(tl.sum(x_2d * x_2d, axis=1) / K) + EPS
        rm += tl.outer(x_rm, x_rm)
    rm = 0.5 * (rm + rm.transpose(1, 0))
    rm = tl.where(tl.arange(None, BLOCK_SIZE_M)[:, None] <= tl.arange(None, BLOCK_SIZE_N)[None, :], rm, 0.0)
    tl.debug_barrier()
    # compute rms
    rms = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        w_mask = k_offset < K
        w_ptr_mask = k_offset * stride_w_K
        w = tl.load(w_ptr + w_ptr_mask, mask=w_mask, other=0)
        w = w.to(tl.float32)
        w = w * w
        rms += tl.sum(w, axis=0)
    rms = tl.sqrt(rms / K) + EPS
    tl.debug_barrier()
    # compute rms_w
    rms_w = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE_N):
        n_offset = n + tl.arange(0, BLOCK_SIZE_N)
        w_mask = n_offset < N
        w_ptr_mask = n_offset * stride_rms_w_N
        w = tl.load(rms_w_ptr + w_ptr_mask, mask=w_mask, other=0)
        w = w.to(tl.float32)
        w = w * w
        rms_w += tl.sum(w, axis=0)
    rms_w = tl.sqrt(rms_w / N) + EPS
    tl.debug_barrier()
    # compute wrms @ x
    pid_m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    pid_n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_mask = pid_m_offset[:, None] < M
    x_ptr_mask = (pid_m_offset * stride_x_M + pid_n_offset[None, :] * stride_x_N)
    x = tl.load(x_ptr + x_ptr_mask, mask=x_mask, other=0)
    x = x.to(tl.float32)
    w_mask = pid_n_offset[:, None] < N
    w_ptr_mask = pid_n_offset * stride_w_N
    w = tl.load(w_ptr + w_ptr_mask, mask=w_mask, other=0)
    w = w.to(tl.float32)
    # normalize
    if USE_FP8:
        x_rms = tl.sqrt(tl.sum(x * x, axis=1) / K) + EPS
        w_rms = tl.sqrt(tl.sum(w * w, axis=0) / K) + EPS
        x_scale = tl.math.rsqrt(K) * tl.math.rsqrt(x_rms)
        w_scale = tl.math.rsqrt(N) * tl.math.rsqrt(w_rms)
        x = x * x_scale[:, None]
        w = w * w_scale[None, :]
        rm = rm * x_scale[:, None] * w_scale[None, :]
    else:
        x_rms = tl.sqrt(tl.sum(x * x, axis=1) / K) + EPS
        w_rms = tl.sqrt(tl.sum(w * w, axis=0) / K) + EPS
        rm = rm * x_rms[:, None] * w_rms[None, :]
    out = tl.dot(x, w, rm)
    tl.debug_barrier()
    # apply rotary embedding
    if RBE_EPILOGUE:
        out = apply_rotary_embedding(out, THETA, start_token_position, M)
    # write back out
    out_mask = (pid_m_offset[:, None] < M) & (pid_n_offset[None, :] < N)
    out_ptr_mask = (pid_m_offset * stride_out_M + pid_n_offset * stride_out_N)[:, None]
    tl.store(out_ptr + out_ptr_mask, out, mask=out_mask)

@triton.jit
def rms_matmul_rbe_qkv(
    q_ptr, k_ptr, v_ptr,
    wq_ptr, wk_ptr, wv_ptr,
    rms_wq_ptr, rms_wk_ptr, rms_wv_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    M, N, K,
    stride_q_batch, stride_q_M, stride_q_K,
    stride_k_batch, stride_k_M, stride_k_K,
    stride_v_batch, stride_v_M, stride_v_K,
    stride_wq_K, stride_wq_N,
    stride_wk_K, stride_wk_N,
    stride_wv_K, stride_wv_N
