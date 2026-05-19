import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d, O, last_kv,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2)
    
    acc = tl.zeros((BLOCK_MODEL_K, BLOCK_MODEL_V), dtype=tl.float32)
    
    if USE_INITIAL_STATE:
        p_init = last_kv + i_h * D_MODEL_K * D_MODEL_V + i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        acc += tl.load(p_init, mask=(tl.arange(0, BLOCK_MODEL_K)[:, None] < BLOCK_MODEL_K) & 
                       (tl.arange(0, BLOCK_MODEL_V)[None, :] < BLOCK_MODEL_V), other=0.0)
    
    for i_b in range(NUM_BLOCK):
        p_S = S + i_h * NUM_BLOCK * D_MODEL_K * D_MODEL_V + i_b * D_MODEL_K * D_MODEL_V + \
              i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        p_d = d + i_h * NUM_BLOCK + i_b
        
        block_S = tl.load(p_S, mask=(tl.arange(0, BLOCK_MODEL_K)[:, None] < BLOCK_MODEL_K) & 
                          (tl.arange(0, BLOCK_MODEL_V)[None, :] < BLOCK_MODEL_V), other=0.0)
        decay = tl.math.exp2(tl.load(p_d))
        
        acc = acc * decay + block_S
        
        p_O = O + i_h * NUM_BLOCK * D_MODEL_K * D_MODEL_V + i_b * D_MODEL_K * D_MODEL_V + \
              i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        tl.store(p_O, acc.to(p_O.dtype.element_ty))

    if STORE_FINAL_STATE:
        p_final = last_kv + i_h * D_MODEL_K * D_MODEL_V + i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        tl.store(p_final, acc.to(p_final.dtype.element_ty))

@triton.jit
def _bwd_recurrence(
    S, d, DI, DG, DL, DS,
    NUM_HEAD, NUM_BLOCK,
    D_MODEL_K, D_MODEL_V,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
    CHECK: tl.constexpr
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2)
    
    grad_acc = tl.zeros((BLOCK_MODEL_K, BLOCK_MODEL_V), dtype=tl.float32)
    
    for i_b in range(NUM_BLOCK-1, -1, -1):
        p_S = S + i_h * NUM_BLOCK * D_MODEL_K * D_MODEL_V + i_b * D_MODEL_K * D_MODEL_V + \
              i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        p_d = d + i_h * NUM_BLOCK + i_b
        
        block_S = tl.load(p_S)
        decay = tl.math.exp2(tl.load(p_d))
        
        p_DS = DS + i_h * NUM_BLOCK * D_MODEL_K * D_MODEL_V + i_b * D_MODEL_K * D_MODEL_V + \
               i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        
        grad_acc += tl.load(p_DS)
        tl.store(p_DS, grad_acc.to(p_DS.dtype.element_ty))
        
        grad_d = tl.sum(grad_acc * block_S) * tl.math.log2(tl.math.e)
        tl.atomic_add(DG + i_h * NUM_BLOCK + i_b, grad_d)
        
        grad_acc = grad_acc * decay

    if CHECK:
        p_DL = DL + i_h * D_MODEL_K * D_MODEL_V + i_k * BLOCK_MODEL_K * D_MODEL_V + i_v * BLOCK_MODEL_V
        tl.store(p_DL, grad_acc.to(p_DL.dtype.element_ty))

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, d, last_kv=None):
        B, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = S.shape
        O = torch.empty_like(S)
        final_state = torch.empty(B, NUM_HEAD, D_MODEL_K, D_MODEL_V, device=S.device) if last_kv is None else None
        
        BLOCK_K = 64
        BLOCK_V = 64
        
        grid = (triton.cdiv(D_MODEL_K, BLOCK_K), 
                triton.cdiv(D_MODEL_V, BLOCK_V), 
                B * NUM_HEAD)
        
        _fwd_recurrence[grid](
            S, d, O, last_kv,
            NUM_HEAD, NUM_BLOCK,
            D_MODEL_K, D_MODEL_V,
            BLOCK_K, BLOCK_V,
            last_kv is not None,
            final_state is not None,
            S.numel() % (BLOCK_K*BLOCK_V) != 0
        )
        
        ctx.save_for_backward(S, d, last_kv)
        return O, final_state

    @staticmethod
    def backward(ctx, DO, Dfinal_state=None):
        S, d, last_kv = ctx.saved_tensors
        B, NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V = S.shape
        
        DI = torch.zeros_like(S)
        DG = torch.zeros_like(d)
        DL = torch.zeros_like(last_kv) if last_kv is not None else None
        DS = torch.zeros_like(S)
        
        BLOCK_K = 64
        BLOCK_V = 64
        
        grid = (triton.cdiv(D_MODEL_K, BLOCK_K), 
                triton.cdiv(D_MODEL_V, BLOCK_V), 
                B * NUM_HEAD)
        
        _bwd_recurrence[grid](
            S, d, DI, DG, DL, DS,
            NUM_HEAD, NUM_BLOCK,
            D_MODEL_K, D_MODEL_V,
            BLOCK_K, BLOCK_V,
            S.numel() % (BLOCK_K*BLOCK_V) != 0
        )
        
        return DS, DG, DL if last_kv is not None else None

# Forward pass
S = torch.randn(2, 8, 16, 256, 512, device='cuda')  # [B, H, N, Dk, Dv]
d = torch.randn(2, 8, 16, device='cuda')             # Decay factors
output, state = ChunkGateRecurrent.apply(S, d)

# Backward pass
grad_output = torch.randn_like(S)
grad_S, grad_d, _ = torch.autograd.grad(output, (S, d), grad_output)
