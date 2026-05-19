import torch
import triton
import triton.language as tl
from flash_attn.ops.triton.k_split import K_SPLITS

@triton.jit
def _int8_matmul_rowwise_dequantize(
    C,
    A,
    B,
    state_x_ptr,
    state_w_ptr,
    bias,
    stride_cm,
    stride_cn,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    stride_xm,
    stride_xn,
    stride_wm,
    stride_wn,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order dimensions for SPLIT_K = 1
    rk = pid % K
    rm = (pid // (SPLIT_K * K)) % grid_m
    rn = (pid // (SPLIT_K * K)) // grid_m
    rp = pid % SPLIT_K
    k_offset = rk * BLOCK_K
    # offset ptrs
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K) + k_offset
    a_ptrs = A + (rm * BLOCK_M + offs_am[:, None]) * stride_am + (offs_k[None, :] * stride_an)
    b_ptrs = B + (offs_k[:, None] * stride_bm + (rn * BLOCK_N + offs_bn[None, :]) * stride_bn)
    # pointers to the c-th chunk of the state
    state_x_ptrs = state_x_ptr + rp * M * N + (rm * BLOCK_M + offs_am[:, None]) * N + (
        rn * BLOCK_N + offs_bn[None, :]
    )
    state_w_ptrs = state_w_ptr + rp * K + (offs_k[None, :] * M * N + (rm * BLOCK_M + offs_am[:, None]) * N + (rn * BLOCK_N + offs_bn[None, :]))
    # initialize acc
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # fetch state
        x = tl.load(state_x_ptrs)
        w = tl.load(state_w_ptrs)
        # scale x; this is the only fp32 operation in the whole grid
        x = tl.libdevice.llrint(x * w)
        # dequantize
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, x).to(tl.int32)
        a_ptrs += BLOCK_K * stride_an
        b_ptrs += BLOCK_K * stride_bm
        state_x_ptrs += BLOCK_K * stride_xm * N
        state_w_ptrs += BLOCK_K * stride_xm * N
    # rematerialize to save registers
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    # load bias
    if bias is not None:
        bias_ptrs = bias + offs_bn[None, :]
        bias_val = tl.load(bias_ptrs)
        acc += bias_val[:, None]
    # scale and store
    c_ptrs = C + stride_cm * (rm * BLOCK_M + offs_am[:, None]) + stride_cn * (rn * BLOCK_N + offs_bn[None, :])
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc)
    else:
        tl.atomic_add(c_ptrs, acc)


def int8_matmul_rowwise_dequantize(
    out,
    a,
    b,
    state_x,
    state_w,
    bias=None,
    a_is_contiguous: bool = False,
    b_is_contiguous: bool = False,
    split_k: int = 1,
):
    assert out.dtype == torch.float16
    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    assert state_x.dtype == torch.float16
    assert state_w.dtype == torch.float16
    assert a.shape[1] == b.shape[0], f"incompatible dimensions, {a.shape=} {b.shape=}"
    M, K = a.shape
    _, N = b.shape

    # handle non-contiguous case by copying
    if not a_is_contiguous:
        a = a.clone()
    if not b_is_contiguous:
        b = b.clone()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * META["SPLIT_K"],)
    with torch.cuda.device(a.device.index):
        _int8_matmul_rowwise_dequantize[grid](
            out,
            a,
            b,
            state_x,
            state_w,
            bias,
            out.stride(0),
            out.stride(1),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            state_x.stride(0),
            state_x.stride(1),
            state_w.stride(0),
            state_w.stride(1),
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            split_k,
        )


@triton.jit
def _int8_matmul_rowwise_dequantize_per_chunk(
    C,
    A,
    B,
    state_x_ptr,
    state_w_ptr,
    bias,
    stride_cm,
    stride_cn,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    stride_xm,
    stride_xn,
    stride_wm,
    stride_wn,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    # chunk k-dim into BLOCK_K chunks
    assert BLOCK_K % CHUNK_K == 0
    # split BLOCK_K into smaller chunks
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order dimensions
    rk = pid % (K // CH
