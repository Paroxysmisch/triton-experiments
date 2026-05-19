import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d, O, last_kv,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K, BLOCK_MODEL_V,
    USE_LAST_KV: tl.constexpr
):
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if head_id >= NUM_HEAD or block_id >= NUM_BLOCK:
        return

    b_h = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    if USE_LAST_KV:
        p_last_kv = tl.make_block_ptr(last_kv, (D_MODEL_K, D_MODEL_V), (D_MODEL_V, 1), (head_id * D_MODEL_K, 0), (BLOCK_MODEL_K, BLOCK_MODEL_V), (1, 0))
        b_h += tl.load(p_last_kv, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, NUM_BLOCK):
        p_S = tl.make_block_ptr(S, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_d = tl.make_block_ptr(d, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_O = tl.make_block_ptr(O, (NUM_BLOCK, D_MODEL_V), (D_MODEL_V, 1), (i, head_id * D_MODEL_V), (1, BLOCK_MODEL_V), (1, 0))

        b_S = tl.load(p_S, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))

        b_O = tl.dot(b_S.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)
        b_h = b_h * tl.math.exp2(b_d) + b_S

        tl.store(p_O, b_O.to(p_O.dtype.element_ty), boundary_check=(0, 1))

    if USE_LAST_KV:
        p_last_kv = tl.make_block_ptr(last_kv, (D_MODEL_K, D_MODEL_V), (D_MODEL_V, 1), (head_id * D_MODEL_K, 0), (BLOCK_MODEL_K, BLOCK_MODEL_V), (1, 0))
        tl.store(p_last_kv, b_h.to(p_last_kv.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def _bwd_recurrence(
    S, d, DI, DG, DL, DS,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K, BLOCK_MODEL_V
):
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    if head_id >= NUM_HEAD or block_id >= NUM_BLOCK:
        return

    b_h = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    for i in range(NUM_BLOCK - 1, -1, -1):
        p_S = tl.make_block_ptr(S, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_d = tl.make_block_ptr(d, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_DI = tl.make_block_ptr(DI, (NUM_BLOCK, D_MODEL_V), (D_MODEL_V, 1), (i, head_id * D_MODEL_V), (1, BLOCK_MODEL_V), (1, 0))
        p_DG = tl.make_block_ptr(DG, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_DL = tl.make_block_ptr(DL, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))
        p_DS = tl.make_block_ptr(DS, (NUM_BLOCK, D_MODEL_K), (D_MODEL_K, 1), (i, head_id * D_MODEL_K), (1, BLOCK_MODEL_K), (1, 0))

        b_S = tl.load(p_S, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_DI = tl.load(p_DI, boundary_check=(0, 1))

        b_DG = tl.dot(b_DI.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)
        b_h = b_h * tl.math.exp2(b_d) + b_S
        b_DS = b_DG * tl.math.exp2(b_d)

        tl.store(p_DG, b_DG.to(p_DG.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_DS, b_DS.to(p_DS.dtype.element_ty), boundary_check=(0, 1))

### High-Level Interface Class

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, d, last_kv=None):
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = S.shape[1], S.shape[0], S.shape[2], S.shape[3]
        BLOCK_MODEL_K, BLOCK_MODEL_V = S.shape[2], S.shape[3]

        O = torch.empty_like(S)
        last_kv_out = torch.empty((NUM_HEAD, D_MODEL_K, D_MODEL_V), device=S.device) if last_kv is not None else None

        USE_LAST_KV = last_kv is not None

        grid = (NUM_HEAD, NUM_BLOCK)
        _fwd_recurrence[grid](
            S, d, O, last_kv,
            NUM_HEAD, NUM_BLOCK,
            D_MODEL_K, D_MODEL_V,
            BLOCK_MODEL_K, BLOCK_MODEL_V,
            USE_LAST_KV
        )

        ctx.save_for_backward(S, d, O, last_kv)
        ctx.num_head = NUM_HEAD
        ctx.num_block = NUM_BLOCK
        ctx.d_model_k = D_MODEL_K
        ctx.d_model_v = D_MODEL_V
        ctx.block_model_k = BLOCK_MODEL_K
        ctx.block_model_v = BLOCK_MODEL_V

        return O, last_kv_out if last_kv_out is not None else O

    @staticmethod
    def backward(ctx, DO):
        S, d, O, last_kv = ctx.saved_tensors
        NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = ctx.num_head, ctx.num_block, ctx.d_model_k, ctx.d_model_v
        BLOCK_MODEL_K, BLOCK_MODEL_V = ctx.block_model_k, ctx.block_model_v

        DI = torch.empty_like(DO)
        DG = torch.empty_like(d)
        DL = torch.empty_like(last_kv) if last_kv is not None else None
        DS = torch.empty_like(S)

        grid = (NUM_HEAD, NUM_BLOCK)
        _bwd_recurrence[grid](
            S, d, DI, DG, DL, DS,
            NUM_HEAD, NUM_BLOCK,
            D_MODEL_K, D_MODEL_V,
            BLOCK_MODEL_K, BLOCK_MODEL_V
        )

        return DS, DG, DL
