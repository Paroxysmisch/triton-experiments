import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,  # tensors
    L,  # tensor
    Out,  # tensor
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh, stride_lm,
    Z, H, N_CTX,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr,
    M_LT_N: tl.constexpr,
    NUM_PATCHES: tl.constexpr,
    rematerialize_offsets: tl.constexpr,
):
    # Triton kernel for computing the forward pass of self-attention.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    if M_LT_N:
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    L_block_ptr = tl.make_block_ptr(
        base=L,
        shape=(N_CTX,),
        strides=(stride_lm,),
        offsets=(start_m * BLOCK_M, ),
        block_shape=(BLOCK_M, ),
        order=(0, ),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    if rematerialize_offsets:
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    else:
        offs_m = tl.arange(0, BLOCK_M)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (N_CTX // BLOCK_N)

    if IS_CAUSAL:
        hi = ((start_m + 1) * BLOCK_M // BLOCK_N)

    for i in range(lo, hi):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        if USE_FP8:
            if i % 2 == 0:
                L_base_ptr = tl.make_block_ptr(
                    base=L + qvk_offset,
                    shape=(N_CTX, ),
                    strides=(stride_lm, ),
                    offsets=(i * BLOCK_N, ),
                    block_shape=(BLOCK_N, ),
                    order=(0, )
                )
                lambda_val = tl.load(L_base_ptr)
                lambda_val = lambda_val.to(tl.float32)
                v = v * lambda_val[:, None]
            else:
                lambda_val = 1.0
        else:
            lambda_val = tl.load(L_block_ptr)
            lambda_val = lambda_val.to(tl.float32)
            if i % 2 == 0:
                v = v * lambda_val[:, None]

        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        if i % 2 == 0:
            if USE_FP8:
                lambda_val = lambda_val.to(tl.float16)
                v = (v * lambda_val[:, None]).to(tl.float16)
            else:
                v = (v * lambda_val[:, None]).to(tl.float16)
            if M_LT_N:
                o = tl.dot(q, v, out_dtype=tl.float32)
                o = o.to(tl.float16)
                tl.store(O_block_ptr, o, boundary_check=(0, 1))
        else:
            if USE_FP8:
                lambda_val = lambda_val.to(tl.float16)
                v = (v / lambda_val[:, None]).to(tl.float16)
            else:
                v = (v / lambda_val[:, None]).to(tl.float16)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if M_LT_N:
            O_block_ptr = tl.advance(O_block_ptr, (BLOCK_N, 0))

    if not M_LT_N:
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    for i in range(hi, N_CTX // BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        if USE_FP8:
            lambda_val = 1.0
        else:
            lambda_val = tl.load(L_block_ptr)
            lambda_val = lambda_val.to(tl.float32)
        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        if i % 2 == 0:
            if USE_FP8:
                lambda_val = lambda_val.to(tl.float16)
                v = v * lambda_val[:, None]
            else:
                v = v * lambda_val[:, None]
            o = tl.dot(q, v, out_dtype=tl.float32)
            o = o.to(tl.float16)
            tl.store(O_block_ptr, o, boundary_check=(0, 1))
        else:
            if USE_FP8:
                lambda_val = lambda_val.to(tl.float16)
                v = v / lambda_val[:, None]
            else:
                v = v / lambda_val[:, None]
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if not M_LT_N:
            O_block_ptr = tl.advance(O_block_ptr, (BLOCK_N, 0))


def triton_fa(
    q, k, v, l, sm_scale, causal: bool, num_patches: int, dtype: torch.dtype, use_fp8: bool
):
    # Wrapper function for invoking the Triton kernel.
    head_size = q.shape[-1]
    batch, head, seq_len, d = q.shape
    assert q.shape == k.shape == v.shape
    assert l.shape == (batch, head, seq_len)
    o = torch.empty_like(q)
    assert (
        seq_len % 32 == 0
    ), "seq_len must be divisible by BLOCK_M (which is 32 throughout this code)"
    assert (
        d == 16 or d == 32 or d == 64 or d == 128 or d == 256 or d == 512
    ), "hidden size must be divisible by BLOCK_DMODEL (which is 64 throughout this code)"
    if use_fp8:
        assert d in [16, 32, 64, 128, 256, 512], "only can use FP8 if hidden size is in [16, 32, 64, 128, 256, 512]"
    BLOCK = 32
    if d >= 256:
        BLOCK = 64
    if d >= 512:
        BLOCK = 128
    num_warps = 4 if d <= 256 else 8

    grid = (triton.cdiv(seq_len, BLOCK), batch * head, 1)
    if num_patches != -1:
        assert num_patches * 2 <= seq_len
        assert num_patches % 3
