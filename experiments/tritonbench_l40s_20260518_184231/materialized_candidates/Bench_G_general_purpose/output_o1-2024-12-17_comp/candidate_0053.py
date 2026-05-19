import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_recurrence(
    S_ptr, d_ptr, O_ptr, LAST_KV_ptr,
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    strideS_h: tl.constexpr, strideS_b: tl.constexpr, strideS_k: tl.constexpr, strideS_v: tl.constexpr,
    strideD_h: tl.constexpr, strideD_b: tl.constexpr, strideD_k: tl.constexpr, strideD_v: tl.constexpr,
    strideO_h: tl.constexpr, strideO_b: tl.constexpr, strideO_k: tl.constexpr, strideO_v: tl.constexpr,
    strideL_h: tl.constexpr, strideL_k: tl.constexpr  # for LAST_KV if needed
):
    # Program id determines which head/block is processed
    pid = tl.program_id(0)
    head_idx = pid // NUM_BLOCK
    block_idx = pid % NUM_BLOCK

    # Base pointers for the current head/block
    S_base_k = S_ptr + head_idx * strideS_h + block_idx * strideS_b
    d_base_k = d_ptr + head_idx * strideD_h + block_idx * strideD_b
    O_base_k = O_ptr + head_idx * strideO_h + block_idx * strideO_b

    # Optionally load from last_kv for recurrent state
    last_kv_base = LAST_KV_ptr + head_idx * strideL_h if LAST_KV_ptr else 0

    # We illustrate K and V dimension updates in two loops for demonstration
    # In practice, these can be vectorized or done in parallel, pulling data in blocks
    for k_offset in range(0, BLOCK_MODEL_K):
        k_val = tl.load(S_base_k + k_offset * strideS_k, mask=True, other=0.0)
        decay = tl.load(d_base_k + k_offset * strideD_k, mask=True, other=1.0)
        # Recurrent update (dummy op here)
        if LAST_KV_ptr:
            prev_kv_val = tl.load(last_kv_base + k_offset * strideL_k, mask=True, other=0.0)
            k_val = k_val + decay * prev_kv_val
        # Store updated K to output
        tl.store(O_base_k + k_offset * strideO_k, k_val, mask=True)

    for v_offset in range(0, BLOCK_MODEL_V):
        v_val = tl.load(S_base_k + v_offset * strideS_v, mask=True, other=0.0)
        decay = tl.load(d_base_k + v_offset * strideD_v, mask=True, other=1.0)
        if LAST_KV_ptr:
            prev_kv_val = tl.load(last_kv_base + v_offset * strideL_k, mask=True, other=0.0)
            v_val = v_val + decay * prev_kv_val
        tl.store(O_base_k + v_offset * strideO_v, v_val, mask=True)


@triton.jit
def _bwd_recurrence(
    S_ptr, d_ptr,
    DI_ptr, DG_ptr, DL_ptr, DS_ptr,
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    strideS_h: tl.constexpr, strideS_b: tl.constexpr, strideS_k: tl.constexpr, strideS_v: tl.constexpr,
    strideD_h: tl.constexpr, strideD_b: tl.constexpr, strideD_k: tl.constexpr, strideD_v: tl.constexpr,
    strideDI_h: tl.constexpr, strideDI_b: tl.constexpr, strideDI_k: tl.constexpr, strideDI_v: tl.constexpr,
    strideDG_h: tl.constexpr, strideDG_b: tl.constexpr, strideDG_k: tl.constexpr, strideDG_v: tl.constexpr,
    strideDL_h: tl.constexpr, strideDL_k: tl.constexpr,
    strideDS_h: tl.constexpr, strideDS_b: tl.constexpr, strideDS_k: tl.constexpr, strideDS_v: tl.constexpr
):
    # Program id determines which head/block is processed
    pid = tl.program_id(0)
    head_idx = pid // NUM_BLOCK
    block_idx = pid % NUM_BLOCK

    # Base pointers
    S_base_k = S_ptr + head_idx * strideS_h + block_idx * strideS_b
    d_base_k = d_ptr + head_idx * strideD_h + block_idx * strideD_b

    DI_base_k = DI_ptr + head_idx * strideDI_h + block_idx * strideDI_b
    DG_base_k = DG_ptr + head_idx * strideDG_h + block_idx * strideDG_b
    DS_base_k = DS_ptr + head_idx * strideDS_h + block_idx * strideDS_b

    DL_base_k = DL_ptr + head_idx * strideDL_h  # remainder for last_kv gradient accumulation

    # Reverse iteration to back-propagate
    for k_offset in range(BLOCK_MODEL_K - 1, -1, -1):
        # Load forward data
        s_val = tl.load(S_base_k + k_offset * strideS_k, mask=True, other=0.0)
        d_val = tl.load(d_base_k + k_offset * strideD_k, mask=True, other=1.0)

        # Mock backward pass computation
        dO_k = tl.load(DI_base_k + k_offset * strideDI_k, mask=True, other=0.0)
        ds_val = dO_k  # Here we treat partial derivative as example
        dd_val = dO_k * s_val

        # Accumulate gradients
        prev_dL = dd_val  # gradient to last_kv
        tl.atomic_add(DL_base_k + k_offset * strideDL_k, prev_dL)
        # Store partial grads
        tl.store(DS_base_k + k_offset * strideDS_k, ds_val, mask=True)
        tl.store(DG_base_k + k_offset * strideDG_k, dd_val, mask=True)

    for v_offset in range(BLOCK_MODEL_V - 1, -1, -1):
        s_val = tl.load(S_base_k + v_offset * strideS_v, mask=True, other=0.0)
        d_val = tl.load(d_base_k + v_offset * strideD_v, mask=True, other=1.0)

        dO_v = tl.load(DI_base_k + v_offset * strideDI_v, mask=True, other=0.0)
        ds_val = dO_v
        dd_val = dO_v * s_val

        prev_dL = dd_val
        tl.atomic_add(DL_base_k + v_offset * strideDL_k, prev_dL)
        tl.store(DS_base_k + v_offset * strideDS_v, ds_val, mask=True)
        tl.store(DG_base_k + v_offset * strideDG_v, dd_val, mask=True)


class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        # Dimensions, assume kv has shape [NUM_HEAD, NUM_BLOCK, D_MODEL] for demonstration
        NUM_HEAD = kv.shape[0]
        NUM_BLOCK = kv.shape[1]
        D_MODEL = kv.shape[2]
        # Example split into K and V
        D_MODEL_K = D_MODEL // 2
        D_MODEL_V = D_MODEL - D_MODEL_K
        BLOCK_MODEL_K = D_MODEL_K
        BLOCK_MODEL_V = D_MODEL_V

        # Allocate output
        out = torch.empty_like(kv)

        # Launch fwd kernel over each (head, block)
        grid = (NUM_HEAD * NUM_BLOCK,)
        # Flatten/reshape for demonstration
        kv_ptr = kv
        d_ptr = cross_decay
        last_kv_ptr = last_kv if last_kv is not None else torch.zeros_like(kv)

        _fwd_recurrence[grid](
            kv_ptr, d_ptr, out, last_kv_ptr,
            NUM_HEAD=NUM_HEAD, NUM_BLOCK=NUM_BLOCK,
            D_MODEL_K=D_MODEL_K, D_MODEL_V=D_MODEL_V,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V,
            strideS_h=kv_ptr.stride(0), strideS_b=kv_ptr.stride(1), strideS_k=1, strideS_v=1,
            strideD_h=d_ptr.stride(0), strideD_b=d_ptr.stride(1), strideD_k=1, strideD_v=1,
            strideO_h=out.stride(0), strideO_b=out.stride(1), strideO_k=1, strideO_v=1,
            strideL_h=last_kv_ptr.stride(0), strideL_k=1
        )

        ctx.save_for_backward(kv, cross_decay, last_kv, out)
        ctx.NUM_HEAD = NUM_HEAD
        ctx.NUM_BLOCK = NUM_BLOCK
        ctx.D_MODEL_K = D_MODEL_K
        ctx.D_MODEL_V = D_MODEL_V
        ctx.BLOCK_MODEL_K = BLOCK_MODEL_K
        ctx.BLOCK_MODEL_V = BLOCK_MODEL_V
        return out

    @staticmethod
    def backward(ctx, dO):
        kv, cross_decay, last_kv, out = ctx.saved_tensors
        NUM_HEAD = ctx.NUM_HEAD
        NUM_BLOCK = ctx.NUM_BLOCK
        D_MODEL_K = ctx.D_MODEL_K
        D_MODEL_V = ctx.D_MODEL_V
        BLOCK_MODEL_K = ctx.BLOCK_MODEL_K
        BLOCK_MODEL_V = ctx.BLOCK_MODEL_V

        # Allocate grads
        dkv = torch.zeros_like(kv)
        dcross_decay = torch.zeros_like(cross_decay)
        dlast_kv = torch.zeros_like(last_kv) if last_kv is not None else None
        dS_tmp = torch.zeros_like(kv)

        grid = (NUM_HEAD * NUM_BLOCK,)

        _bwd_recurrence[grid](
            kv, cross_decay,
            dO, dcross_decay, dlast_kv if dlast_kv is not None else torch.zeros_like(kv), dS_tmp,
            NUM_HEAD=NUM_HEAD, NUM_BLOCK=NUM_BLOCK,
            D_MODEL_K=D_MODEL_K, D_MODEL_V=D_MODEL_V,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V,
            strideS_h=kv.stride(0), strideS_b=kv.stride(1), strideS_k=1, strideS_v=1,
            strideD_h=cross_decay.stride(0), strideD_b=cross_decay.stride(1), strideD_k=1, strideD_v=1,
            strideDI_h=dO.stride(0), strideDI_b=dO.stride(1), strideDI_k=1, strideDI_v=1,
            strideDG_h=dcross_decay.stride(0), strideDG_b=dcross_decay.stride(1), strideDG_k=1, strideDG_v=1,
            strideDL_h=dlast_kv.stride(0) if dlast_kv is not None else 1, strideDL_k=1,
            strideDS_h=dS_tmp.stride(0), strideDS_b=dS_tmp.stride(1), strideDS_k=1, strideDS_v=1
        )

        # dS_tmp holds partial gradient wrt kv
        dkv += dS_tmp
        return dkv, dcross_decay, dlast_kv if last_kv is not None else None


def chunk_gate_recurrent_forward(kv, cross_decay, last_kv=None):
    return ChunkGateRecurrent.apply(kv, cross_decay, last_kv)


def chunk_gate_recurrent_backward(dO, ctx):
    return ctx.backward(dO)
