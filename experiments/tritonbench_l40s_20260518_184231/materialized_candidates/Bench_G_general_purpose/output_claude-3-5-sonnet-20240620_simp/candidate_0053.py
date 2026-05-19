import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_recurrence(
    S_ptr, d_ptr, O_ptr, 
    last_kv_ptr,
    NUM_HEAD: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr,
    D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
    stride_s_h, stride_s_b, stride_s_k,
    stride_d_h, stride_d_b,
    stride_o_h, stride_o_b, stride_o_v,
    stride_lkv_h, stride_lkv_k
):
    # Get program ID
    pid_h = tl.program_id(0)  # Head dimension
    pid_b = tl.program_id(1)  # Batch dimension
    
    # Compute base pointers
    s_offset = pid_h * stride_s_h + pid_b * stride_s_b
    d_offset = pid_h * stride_d_h + pid_b * stride_d_b
    o_offset = pid_h * stride_o_h + pid_b * stride_o_b
    lkv_offset = pid_h * stride_lkv_h
    
    # Load last key-value if available
    last_k = tl.zeros([D_MODEL_K], dtype=tl.float32)
    if last_kv_ptr:
        last_k = tl.load(last_kv_ptr + lkv_offset + tl.arange(0, D_MODEL_K) * stride_lkv_k)
    
    # Initialize accumulators
    acc_v = tl.zeros([D_MODEL_V], dtype=tl.float32)
    
    # Forward pass through blocks
    for block in range(NUM_BLOCK):
        # Load decay factor
        d = tl.load(d_ptr + d_offset + block)
        
        # Load current block's key
        k_idx = tl.arange(0, BLOCK_MODEL_K)
        k = tl.load(S_ptr + s_offset + block * BLOCK_MODEL_K + k_idx)
        
        # Compute attention scores
        scores = tl.sum(k * last_k) / tl.sqrt(D_MODEL_K)
        scores = tl.sigmoid(scores)
        
        # Load and accumulate values
        v_idx = tl.arange(0, BLOCK_MODEL_V)
        v = tl.load(S_ptr + s_offset + block * BLOCK_MODEL_V + v_idx + D_MODEL_K)
        acc_v = acc_v * d + v * scores
        
        # Update last key
        last_k = k
    
    # Store final output
    o_idx = tl.arange(0, D_MODEL_V)
    tl.store(O_ptr + o_offset + o_idx * stride_o_v, acc_v)

@triton.jit
def _bwd_recurrence(
    S_ptr, d_ptr,
    DI_ptr, DG_ptr, DL_ptr, DS_ptr,
    NUM_HEAD: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr,
    D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
    stride_s_h, stride_s_b, stride_s_k,
    stride_d_h, stride_d_b,
    stride_di_h, stride_di_b, stride_di_v,
    stride_dg_h, stride_dg_b,
    stride_dl_h, stride_dl_k,
    stride_ds_h, stride_ds_b
):
    # Get program ID
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Compute base pointers
    s_offset = pid_h * stride_s_h + pid_b * stride_s_b
    d_offset = pid_h * stride_d_h + pid_b * stride_d_b
    di_offset = pid_h * stride_di_h + pid_b * stride_di_b
    dg_offset = pid_h * stride_dg_h + pid_b * stride_dg_b
    dl_offset = pid_h * stride_dl_h
    ds_offset = pid_h * stride_ds_h + pid_b * stride_ds_b
    
    # Initialize gradients
    d_last_k = tl.zeros([D_MODEL_K], dtype=tl.float32)
    d_acc_v = tl.zeros([D_MODEL_V], dtype=tl.float32)
    
    # Backward pass through blocks
    for block in range(NUM_BLOCK-1, -1, -1):
        # Load decay factor and gradients
        d = tl.load(d_ptr + d_offset + block)
        
        # Load current block's key and value
        k_idx = tl.arange(0, BLOCK_MODEL_K)
        k = tl.load(S_ptr + s_offset + block * BLOCK_MODEL_K + k_idx)
        
        v_idx = tl.arange(0, BLOCK_MODEL_V)
        v = tl.load(S_ptr + s_offset + block * BLOCK_MODEL_V + v_idx + D_MODEL_K)
        
        # Compute attention scores gradient
        scores = tl.sum(k * d_last_k) / tl.sqrt(D_MODEL_K)
        scores_grad = tl.sigmoid(scores) * (1 - tl.sigmoid(scores))
        
        # Update gradients
        d_k = d_last_k * scores_grad / tl.sqrt(D_MODEL_K)
        d_v = d_acc_v * tl.sigmoid(scores)
        
        # Store gradients
        tl.store(DI_ptr + di_offset + v_idx * stride_di_v + block * D_MODEL_V, d_v)
        tl.store(DG_ptr + dg_offset + k_idx + block * D_MODEL_K, d_k)
        
        # Update accumulated gradients
        d_acc_v = d_acc_v * d
        d_last_k = k
    
    # Store last key gradients
    tl.store(DL_ptr + dl_offset + tl.arange(0, D_MODEL_K) * stride_dl_k, d_last_k)

class ChunkGateRecurrent:
    def __init__(self, num_head, num_block, d_model_k, d_model_v):
        self.num_head = num_head
        self.num_block = num_block
        self.d_model_k = d_model_k
        self.d_model_v = d_model_v
        self.block_model_k = d_model_k // num_block
        self.block_model_v = d_model_v // num_block

    def forward(self, S, d, last_kv=None):
        batch_size = S.shape[0]
        O = torch.zeros((batch_size, self.num_head, self.d_model_v), 
                       device=S.device, dtype=S.dtype)
        
        # Launch kernel
        grid = (self.num_head, batch_size)
        _fwd_recurrence[grid](
            S, d, O, last_kv,
            self.num_head, self.num_block,
            self.d_model_k, self.d_model_v,
            self.block_model_k, self.block_model_v,
            S.stride(1), S.stride(0), S.stride(2),
            d.stride(1), d.stride(0),
            O.stride(1), O.stride(0), O.stride(2),
            last_kv.stride(1) if last_kv is not None else 0,
            last_kv.stride(2) if last_kv is not None else 0
        )
        return O

    def backward(self, S, d, DI, DG, DL, DS):
        batch_size = S.shape[0]
        
        # Launch kernel
        grid = (self.num_head, batch_size)
        _bwd_recurrence[grid](
            S, d, DI, DG, DL, DS,
            self.num_head, self.num_block,
            self.d_model_k, self.d_model_v,
            self.block_model_k, self.block_model_v,
            S.stride(1), S.stride(0), S.stride(2),
            d.stride(1), d.stride(0),
            DI.stride(1), DI.stride(0), DI.stride(2),
            DG.stride(1), DG.stride(0),
            DL.stride(1), DL.stride(2),
            DS.stride(1), DS.stride(0)
        )
        return DI, DG, DL
