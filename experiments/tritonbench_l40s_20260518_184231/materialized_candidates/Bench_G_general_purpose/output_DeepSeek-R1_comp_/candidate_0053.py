import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S, d, O, last_kv,
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    S_BLOCK_STRIDE, S_HEAD_STRIDE, S_K_STRIDE, S_V_STRIDE,
    d_HEAD_STRIDE, d_K_STRIDE, d_BLOCK_STRIDE,
    O_BLOCK_STRIDE, O_HEAD_STRIDE, O_K_STRIDE, O_V_STRIDE,
    last_kv_HEAD_STRIDE, last_kv_K_STRIDE, last_kv_V_STRIDE,
):
    pid_head = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)
    
    k_offset = pid_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)
    v_offset = pid_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)
    
    mask_k = k_offset < D_MODEL_K
    mask_v = v_offset < D_MODEL_V
    mask = mask_k[:, None] & mask_v[None, :]
    
    S_ptr = S + pid_head * S_HEAD_STRIDE + (k_offset[:, None] * S_K_STRIDE + v_offset[None, :] * S_V_STRIDE)
    d_ptr = d + pid_head * d_HEAD_STRIDE + k_offset * d_K_STRIDE
    O_ptr = O + pid_head * O_HEAD_STRIDE + (k_offset[:, None] * O_K_STRIDE + v_offset[None, :] * O_V_STRIDE)
    last_kv_ptr = last_kv + pid_head * last_kv_HEAD_STRIDE + (k_offset[:, None] * last_kv_K_STRIDE + v_offset[None, :] * last_kv_V_STRIDE)
    
    if last_kv is not None:
        acc = tl.load(last_kv_ptr, mask=mask, other=0.0)
    else:
        acc = tl.zeros((BLOCK_MODEL_K, BLOCK_MODEL_V), dtype=tl.float32)
    
    for block_idx in range(NUM_BLOCK):
        curr_S_ptr = S_ptr + block_idx * S_BLOCK_STRIDE
        curr_d_ptr = d_ptr + block_idx * d_BLOCK_STRIDE
        curr_O_ptr = O_ptr + block_idx * O_BLOCK_STRIDE
        
        s = tl.load(curr_S_ptr, mask=mask, other=0.0)
        d_val = tl.load(curr_d_ptr, mask=mask_k, other=0.0)
        d_val = d_val[:, None]
        
        acc = acc * d_val + s
        tl.store(curr_O_ptr, acc, mask=mask)

@triton.jit
def _bwd_recurrence(
    S, d, O, last_kv, DI, DG, DL,
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    S_BLOCK_STRIDE, S_HEAD_STRIDE, S_K_STRIDE, S_V_STRIDE,
    d_HEAD_STRIDE, d_K_STRIDE, d_BLOCK_STRIDE,
    O_BLOCK_STRIDE, O_HEAD_STRIDE, O_K_STRIDE, O_V_STRIDE,
    last_kv_HEAD_STRIDE, last_kv_K_STRIDE, last_kv_V_STRIDE,
    DI_BLOCK_STRIDE, DI_HEAD_STRIDE, DI_K_STRIDE, DI_V_STRIDE,
    DG_HEAD_STRIDE, DG_K_STRIDE, DG_BLOCK_STRIDE,
    DL_HEAD_STRIDE, DL_K_STRIDE, DL_V_STRIDE,
):
    pid_head = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)
    
    k_offset = pid_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)
    v_offset = pid_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)
    
    mask_k = k_offset < D_MODEL_K
    mask_v = v_offset < D_MODEL_V
    mask = mask_k[:, None] & mask_v[None, :]
    
    S_ptr = S + pid_head * S_HEAD_STRIDE + (k_offset[:, None] * S_K_STRIDE + v_offset[None, :] * S_V_STRIDE)
    d_ptr = d + pid_head * d_HEAD_STRIDE + k_offset * d_K_STRIDE
    O_ptr = O + pid_head * O_HEAD_STRIDE + (k_offset[:, None] * O_K_STRIDE + v_offset[None, :] * O_V_STRIDE)
    DI_ptr = DI + pid_head * DI_HEAD_STRIDE + (k_offset[:, None] * DI_K_STRIDE + v_offset[None, :] * DI_V_STRIDE)
    DG_ptr = DG + pid_head * DG_HEAD_STRIDE + k_offset * DG_K_STRIDE
    DL_ptr = DL + pid_head * DL_HEAD_STRIDE + (k_offset[:, None] * DL_K_STRIDE + v_offset[None, :] * DL_V_STRIDE)
    last_kv_ptr = last_kv + pid_head * last_kv_HEAD_STRIDE + (k_offset[:, None] * last_kv_K_STRIDE + v_offset[None, :] * last_kv_V_STRIDE)
    
    grad_acc = tl.zeros((BLOCK_MODEL_K, BLOCK_MODEL_V), dtype=tl.float32)
    
    for block_idx in range(NUM_BLOCK - 1, -1, -1):
        curr_DI_ptr = DI_ptr + block_idx * DI_BLOCK_STRIDE
        curr_S_ptr = S_ptr + block_idx * S_BLOCK_STRIDE
        curr_d_ptr = d_ptr + block_idx * d_BLOCK_STRIDE
        curr_O_ptr = O_ptr + block_idx * O_BLOCK_STRIDE
        
        do = tl.load(curr_O_ptr, mask=mask, other=0.0)
        current_grad = do + grad_acc
        tl.store(curr_DI_ptr, current_grad, mask=mask)
        
        if block_idx == 0:
            o_prev = tl.load(last_kv_ptr, mask=mask, other=0.0)
        else:
            o_prev_ptr = O_ptr + (block_idx - 1) * O_BLOCK_STRIDE
            o_prev = tl.load(o_prev_ptr, mask=mask, other=0.0)
        
        dg_val = o_prev * current_grad
        dg = tl.sum(dg_val, axis=1)
        tl.atomic_add(DG_ptr + block_idx * DG_BLOCK_STRIDE, dg, mask=mask_k)
        
        d_val = tl.load(curr_d_ptr, mask=mask_k, other=0.0)
        grad_acc = d_val[:, None] * current_grad
    
    tl.store(DL_ptr, grad_acc, mask=mask)

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        kv = kv.contiguous()
        cross_decay = cross_decay.contiguous()
        if last_kv is not None:
            last_kv = last_kv.contiguous()
        else:
            last_kv = torch.zeros(kv.size(1), kv.size(2), kv.size(3), dtype=kv.dtype, device=kv.device)
        
        O = torch.empty_like(kv)
        NUM_BLOCK, NUM_HEAD, D_MODEL_K, D_MODEL_V = kv.shape
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 64
        
        grid = (NUM_HEAD, triton.cdiv(D_MODEL_K, BLOCK_MODEL_K), triton.cdiv(D_MODEL_V, BLOCK_MODEL_V))
        
        _fwd_recurrence[grid](
            kv, cross_decay, O, last_kv,
            NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V,
            BLOCK_MODEL_K, BLOCK_MODEL_V,
            kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
            cross_decay.stride(0), cross_decay.stride(1), cross_decay.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            last_kv.stride(0), last_kv.stride(1), last_kv.stride(2),
        )
        
        ctx.save_for_backward(kv, cross_decay, O, last_kv)
        ctx.last_kv_was_none = (last_kv is None)
        return O

    @staticmethod
    def backward(ctx, DO):
        kv, cross_decay, O, last_kv = ctx.saved_tensors
        DO = DO.contiguous()
        
        DI = torch.empty_like(kv)
        DG = torch.zeros_like(cross_decay)
        DL = torch.empty_like(last_kv)
        
        NUM_BLOCK, NUM_HEAD, D_MODEL_K, D_MODEL_V = kv.shape
        BLOCK_MODEL_K = 64
        BLOCK_MODEL_V = 64
        
        grid = (NUM_HEAD, triton.cdiv(D_MODEL_K, BLOCK_MODEL_K), triton.cdiv(D_MODEL_V, BLOCK_MODEL_V))
        
        _bwd_recurrence[grid](
            kv, cross_decay, O, last_kv, DI, DG, DL,
            NUM_HEAD, NUM_BLOCK, D_MODEL_K, D_MODEL_V,
            BLOCK_MODEL_K, BLOCK_MODEL_V,
            kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
            cross_decay.stride(0), cross_decay.stride(1), cross_decay.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            last_kv.stride(0), last_kv.stride(1), last_kv.stride(2),
            DI.stride(0), DI.stride(1), DI.stride(2), DI.stride(3),
            DG.stride(0), DG.stride(1), DG.stride(2),
            DL.stride(0), DL.stride(1), DL.stride(2),
        )
        
        grad_last_kv = DL if not ctx.last_kv_was_none else None
        return DI, DG, grad_last_kv
