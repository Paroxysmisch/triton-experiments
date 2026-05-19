import triton
import triton.language as tl
import torch
from torch.autograd import Function

class Fast_RoPE_Embedding(Function):
    @staticmethod
    def forward(ctx, Q, cos, CQ, FQ, TQ, SQ):
        Q = Q.transpose(1, 2)
        batch, seq, dim = Q.size()
        nheads = dim // 128
        assert(seq <= cos.size(0))
        BLOCK_SIZE, num_warps = 32, 8
        div, mod = divmod(nheads, 4)
        n_groups = div + (mod != 0)
        
        Q = Q.reshape(batch * seq, nheads * 128)
        n_rows, n_cols = Q.size()
        _rope_embedding[(n_rows, n_groups,)](
            Q,       Q.stride(0),
            cos, cos.stride(0),
            CQ, CQ.stride(0),
            FQ, FQ.stride(0),
            TQ, TQ.stride(0),
            SQ, SQ.stride(0),
            seq,
            128, nheads,
            BACKWARD_PASS=False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        Q = Q.reshape(batch, seq, nheads, 128)
        Q = Q.transpose(1, 2)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(Q, cos)
        return Q 

    @staticmethod
    def backward(ctx, dY):
        dY = dY.transpose(1, 2)
        batch, seq, dim = dY.size()
        nheads = dim // 128
        dY = dY.reshape(batch * seq, nheads * 128)
        n_rows, n_cols = dY.size()
        Q, cos = ctx.saved_tensors
        
        BLOCK_SIZE, num_warps = ctx.BLOCK_SIZE, ctx.num_warps
        div, mod = divmod(nheads, 4)
        n_groups = div + (mod != 0)
        _rope_embedding[(n_rows, n_groups,)](
            dY,       dY.stride(0),
            cos, cos.stride(0),
            Q,   Q.stride(0),
            None, None,
            None, None,
            seq,
            128, nheads,
            BACKWARD_PASS=True,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        dY = dY.reshape(batch, seq, nheads, 128)
        dY = dY.transpose(1, 2)
        return dY, None, None, None, None, None

@triton.jit
def _rope_embedding(
    Q,
    cos, 
    CQ,     FQ,     TQ,     SQ,
    seqlen,  
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    start_head = tl.program_id(0) * 4
    head_offsets = tl.arange(0, 4)
    head_ids = start_head + head_offsets
    mask = head_ids < n_heads

    row_position  = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)

    cos_ptrs = cos + row_position*1 + head_ids*2 + cols
    CQ_ptrs = CQ + row_position*1 + head_ids*2 + cols
    FQ_ptrs = FQ + row_position*1 + head_ids*2 + cols
    TQ_ptrs = TQ + row_position*1 + head_ids*2 + cols
    SQ_ptrs = SQ + row_position*1 + head_ids*2 + cols
    Q_ptrs = Q + row_position*n_heads*128 + head_ids*128 + cols
    
    cos1 = tl.load(cos_ptrs, mask = mask, other = 0)
    CQ1 = tl.load(CQ_ptrs, mask = mask, other = 0)
    FQ1 = tl.load(FQ_ptrs, mask = mask, other = 0)
    TQ1 = tl.load(TQ_ptrs, mask = mask, other = 0)
    SQ1 = tl.load(SQ_ptrs, mask = mask, other = 0)
    Q1 = tl.load(Q_ptrs, mask = mask, other = 0).to(cos1.dtype)
        
    if BACKWARD_PASS:
        cos1 = -cos1

    tl.store(cos_ptrs, cos1, mask = mask)
    tl.store(CQ_ptrs, cos1, mask = mask)
    tl.store(FQ_ptrs, cos1, mask = mask)
    tl.store(TQ_ptrs, cos1, mask = mask)
    tl.store(SQ_ptrs, cos1, mask = mask)
    tl.store(Q_ptrs, Q1*cos1, mask = mask)

def fast_rope_embedding(Q, cos, CQ, FQ, TQ, SQ):
    return Fast_RoPE_Embedding.apply(Q, cos, CQ, FQ, TQ, SQ)
