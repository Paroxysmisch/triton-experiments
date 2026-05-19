import triton
import triton.language as tl
import torch

# -------------------------
# Helper functions
# -------------------------

@triton.jit
def _gelu_none(x):
    # exact GELU
    # 0.5 * x * (1.0 + erf(x / sqrt(2)))
    one = 1.0
    half = 0.5
    rsqrt2 = 0.70710678118654752440
    return half * x * (one + tl.erf(x * rsqrt2))

@triton.jit
def _gelu_tanh(x):
    # tanh approximation of GELU
    # 0.5 * x * (1.0 + tanh(sqrt(2/pi)*(x+0.044715x^3)))
    one = 1.0
    half = 0.5
    k0 = 0.044715
    k1 = 0.7978845608  # sqrt(2/pi)
    x_cubed = x * x * x
    inner = k1 * (x + k0 * x_cubed)
    return half * x * (one + tl.tanh(inner))

@triton.jit
def _philox_rand_32(seed, seed_offset, idx):
    # Very simple Philox-like approach: (not a cryptographically secure RNG)
    # The goal here is just to get some pseudo-random bits per element index.
    # Actual Philox may be more involved, but we'll keep it simple for demonstration.
    val = (seed ^ (idx * 0x9E3779B9)) + seed_offset
    # mix bits
    val ^= (val >> 13)
    val *= 0x85ebca6b
    val ^= (val >> 13)
    return tl.float32(val & 0xFFFFFFFF) * (1.0 / 4294967296.0)

@triton.jit
def _fused_bmm_dropout_gelu_kernel(
    X_ptr, Y_ptr, Out_ptr,
    B, N, M, P,
    stride_xb, stride_xn, stride_xm,
    stride_yb, stride_ym, stride_yp,
    stride_ob, stride_on, stride_op,
    p, is_training, use_tanh_gelu,
    seed, seed_offset,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program_id is a 2D mapping: (batch*NBlocks, PBlocks)
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)

    # batch index and block-row index
    b_idx = pid_n // ((N + BLOCK_M - 1) // BLOCK_M)
    n_idx = pid_n % ((N + BLOCK_M - 1) // BLOCK_M)
    p_idx = pid_p

    # row and col offsets
    row_offs = n_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = p_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # clamp within bounds
    row_mask = row_offs < N
    col_mask = col_offs < P

    # batch offset for X, Y, Out
    X_ptr = X_ptr + b_idx * stride_xb
    Y_ptr = Y_ptr + b_idx * stride_yb
    Out_ptr = Out_ptr + b_idx * stride_ob

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k-range in steps of BLOCK_K
    k_blocks = (M + BLOCK_K - 1) // BLOCK_K

    for k in range(k_blocks):
        # K offsets
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < M

        # load X and Y
        # shape of x_tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (row_offs[:, None] * stride_xn + k_offs[None, :] * stride_xm)
        x_mask = row_mask[:, None] & k_mask[None, :]
        x_tile = tl.where(x_mask, tl.load(x_ptrs, mask=x_mask, other=0.0), 0.0)

        # shape of y_tile: [BLOCK_K, BLOCK_N]
        y_ptrs = Y_ptr + (k_offs[:, None] * stride_ym + col_offs[None, :] * stride_yp)
        y_mask = k_mask[:, None] & col_mask[None, :]
        y_tile = tl.where(y_mask, tl.load(y_ptrs, mask=y_mask, other=0.0), 0.0)

        # compute matmul
        acc += tl.dot(x_tile, y_tile)

    # apply dropout + gelu
    # each thread block processes BLOCK_M*BLOCK_N elements
    # we use philox to get a random for each output element
    rid_base = b_idx * (N * P) + (n_idx * BLOCK_M) * P + (p_idx * BLOCK_N)
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if row_mask[i] & col_mask[j]:
                val = acc[i, j]
                if is_training != 0:
                    idx = rid_base + i * P + j
                    r = _philox_rand_32(seed, seed_offset, idx)
                    keep = r > p
                    if keep:
                        val = val / (1.0 - p)
                    else:
                        val = 0.0
                # GELU
                if use_tanh_gelu != 0:
                    val = _gelu_tanh(val)
                else:
                    val = _gelu_none(val)
                acc[i, j] = val

    # store result
    out_ptrs = Out_ptr + (row_offs[:, None] * stride
