import triton
import triton.language as tl

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    q_ptr, k_ptr, v_ptr, h_ptr, h_init_ptr, T, K, V, BT, BK, BV,
    stride_qt, stride_qk, stride_kt, stride_kk, stride_vt, stride_vk, stride_ht, stride_hv,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_V: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(T, BLOCK_SIZE_T)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_k = tl.cdiv(V, BLOCK_SIZE_V)
    num_pid_in_warp = BT * BK * BV
    warp_id = pid // num_pid_in_warp
    pid_m = (pid % num_pid_in_warp) // (BK * BV)
    pid_n = (pid % (BK * BV)) // BV
    pid_k = (pid % (BK * BV)) % BV

    offs_m = pid_m * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_k = pid_k * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    q = tl.load(q_ptr + (offs_m[:, None] * stride_qt + offs_n[None, :] * stride_qk))
    k = tl.load(k_ptr + (offs_m[:, None] * stride_kt + offs_n[None, :] * stride_kk))
    v = tl.load(v_ptr + (offs_m[:, None] * stride_vt + offs_n[None, :] * stride_vk))

    h = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    if h_init_ptr is not None:
        h_init = tl.load(h_init_ptr + (offs_m[:, None] * stride_ht + offs_k[None, :] * stride_hv))
        h += h_init

    for t in range(T):
        q_t = tl.load(q_ptr + (t * stride_qt + offs_n[None, :] * stride_qk))
        k_t = tl.load(k_ptr + (t * stride_kt + offs_n[None, :] * stride_kk))
        v_t = tl.load(v_ptr + (t * stride_vt + offs_n[None, :] * stride_vk))
        h += tl.dot(q_t, k_t) * v_t

    tl.store(h_ptr + (offs_m[:, None] * stride_ht + offs_k[None, :] * stride_hv), h)

@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q_ptr, k_ptr, h_ptr, o_ptr, T, K, V, BT, BK, BV,
    stride_qt, stride_qk, stride_kt, stride_kk, stride_ht, stride_hv, stride_ot, stride_ov,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_V: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(T, BLOCK_SIZE_T)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_k = tl.cdiv(V, BLOCK_SIZE_V)
    num_pid_in_warp = BT * BK * BV
    warp_id = pid // num_pid_in_warp
    pid_m = (pid % num_pid_in_warp) // (BK * BV)
    pid_n = (pid % (BK * BV)) // BV
    pid_k = (pid % (BK * BV)) % BV

    offs_m = pid_m * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_k = pid_k * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    q = tl.load(q_ptr + (offs_m[:, None] * stride_qt + offs_n[None, :] * stride_qk))
    k = tl.load(k_ptr + (offs_m[:, None] * stride_kt + offs_n[None, :] * stride_kk))
    h = tl.load(h_ptr + (offs_m[:, None] * stride_ht + offs_k[None, :] * stride_hv))

    o = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    for t in range(T):
        q_t = tl.load(q_ptr + (t * stride_qt + offs_n[None, :] * stride_qk))
        k_t = tl.load(k_ptr + (t * stride_kt + offs_n[None, :] * stride_kk))
        h_t = tl.load(h_ptr + (t * stride_ht + offs_k[None, :] * stride_hv))
        attn = tl.softmax(tl.dot(q_t, k_t))
        o += attn * h_t

    tl.store(o_ptr + (offs_m[:, None] * stride_ot + offs_k[None, :] * stride_ov), o)

@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q_ptr, k_ptr, o_grad_ptr, h_grad_ptr, T, K, V, BT, BK, BV,
    stride_qt, stride_qk, stride_kt, stride_kk, stride_ogt, stride_ogv, stride_hgt, stride_hgv,
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_V: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(T, BLOCK_SIZE_T)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_k = tl.cdiv(V, BLOCK_SIZE_V)
    num_pid_in_warp = BT * BK * BV
    warp_id = pid // num_pid_in_warp
    pid_m = (pid % num_pid_in_warp) // (BK * BV)
    pid_n = (pid % (BK * BV)) // BV
    pid_k = (pid % (BK * BV)) % BV

    offs_m = pid_m * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_k = pid_k * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    q = tl.load(q_ptr + (offs_m[:, None] * stride_qt + offs_n[None, :] * stride_qk))
    k = tl.load(k_ptr + (offs_m[:, None] * stride_kt + offs_n[None, :] * stride_kk))
    o_grad = tl.load(o_grad_ptr + (offs_m[:, None] * stride_ogt + offs_k[None, :] * stride_ogv))

    h_grad = tl.zeros((BLOCK_SIZE_T, BLOCK_SIZE_V), dtype=tl.float32)
    for t in range(T):
        q_t = tl.load(q_ptr + (t * stride_qt + offs_n[None, :] * stride_qk))
        k_t = tl.load(k_ptr + (t * stride_kt + offs_n[None, :] * stride_kk))
        attn = tl.softmax(tl.dot(q_t, k_t))
        h_grad += o_grad * attn

    tl.store(h_grad_ptr + (offs_m[:, None] * stride_hgt + offs_k[None, :] * stride_hgv), h_grad)
