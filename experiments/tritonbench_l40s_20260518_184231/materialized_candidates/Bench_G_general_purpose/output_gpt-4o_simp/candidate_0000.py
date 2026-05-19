import torch
import triton
import triton.language as tl

# Forward kernel
@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr, stride_q, stride_k, stride_v, stride_out, n_heads, head_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate the starting position of each block
    q_start = pid * BLOCK_SIZE
    k_start = 0
    
    # Load Q and K blocks
    Q = tl.load(Q_ptr + q_start * stride_q + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
    K = tl.load(K_ptr + k_start * stride_k + tl.arange(0, BLOCK_SIZE)[None, :] * head_dim)
    
    # Compute QK^T
    QK = tl.dot(Q, K)
    
    # Softmax along the last dimension
    QK = tl.softmax(QK, axis=-1)
    
    # Load V block
    V = tl.load(V_ptr + k_start * stride_v + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
    
    # Compute the output
    Out = tl.dot(QK, V)
    
    # Store the result
    tl.store(Out_ptr + q_start * stride_out + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim, Out)

# Backward intra-block kernel
@triton.jit
def _bwd_intra_kernel(GradOut_ptr, Q_ptr, K_ptr, V_ptr, GradQ_ptr, GradK_ptr, GradV_ptr, stride_gradout, stride_q, stride_k, stride_v, stride_gradq, stride_gradk, stride_gradv, n_heads, head_dim, BLOCK_SIZE: tl.constexpr, CBLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate the starting position of each block
    q_start = pid * BLOCK_SIZE
    k_start = 0
    
    # Load GradOut, Q, K, V blocks
    GradOut = tl.load(GradOut_ptr + q_start * stride_gradout + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
    Q = tl.load(Q_ptr + q_start * stride_q + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
    K = tl.load(K_ptr + k_start * stride_k + tl.arange(0, BLOCK_SIZE)[None, :] * head_dim)
    V = tl.load(V_ptr + k_start * stride_v + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
    
    # Compute gradients
    GradQ = tl.dot(GradOut, V.T)
    GradK = tl.dot(Q.T, GradOut)
    GradV = tl.dot(GradOut.T, Q)
    
    # Store gradients
    tl.store(GradQ_ptr + q_start * stride_gradq + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim, GradQ)
    tl.store(GradK_ptr + k_start * stride_gradk + tl.arange(0, BLOCK_SIZE)[None, :] * head_dim, GradK)
    tl.store(GradV_ptr + k_start * stride_gradv + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim, GradV)

# Backward inter-block kernel
@triton.jit
def _bwd_inter_kernel(GradOut_ptr, Q_ptr, K_ptr, V_ptr, GradK_ptr, GradV_ptr, stride_gradout, stride_q, stride_k, stride_v, stride_gradk, stride_gradv, n_heads, head_dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Iterate over blocks to accumulate gradients
    for k_start in range(0, head_dim, BLOCK_SIZE):
        # Load blocks
        GradOut = tl.load(GradOut_ptr + pid * stride_gradout + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
        Q = tl.load(Q_ptr + pid * stride_q + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
        K = tl.load(K_ptr + k_start * stride_k + tl.arange(0, BLOCK_SIZE)[None, :] * head_dim)
        V = tl.load(V_ptr + k_start * stride_v + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim)
        
        # Compute and accumulate gradients
        GradK = tl.dot(Q.T, GradOut)
        GradV = tl.dot(GradOut.T, Q)
        
        # Accumulate gradients
        tl.atomic_add(GradK_ptr + k_start * stride_gradk + tl.arange(0, BLOCK_SIZE)[None, :] * head_dim, GradK)
        tl.atomic_add(GradV_ptr + k_start * stride_gradv + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim, GradV)

# Wrapper class
class LightningAttention2NoDecay:
    @staticmethod
    def forward(Q, K, V):
        # Define output tensor
        Out = torch.empty_like(Q)
        
        # Define grid size
        grid = (Q.shape[0] // 64, )
        
        # Launch forward kernel
        _fwd_kernel[grid](Q, K, V, Out, Q.stride(0), K.stride(0), V.stride(0), Out.stride(0), Q.shape[1], Q.shape[2], BLOCK_SIZE=64)
        
        return Out

    @staticmethod
    def backward(GradOut, Q, K, V):
        # Define gradient tensors
        GradQ = torch.empty_like(Q)
        GradK = torch.empty_like(K)
        GradV = torch.empty_like(V)
        
        # Define grid size
        grid = (Q.shape[0] // 64, )
        
        # Launch backward intra-block kernel
        _bwd_intra_kernel[grid](GradOut, Q, K, V, GradQ, GradK, GradV, GradOut.stride(0), Q.stride(0), K.stride(0), V.stride(0), GradQ.stride(0), GradK.stride(0), GradV.stride(0), Q.shape[1], Q.shape[2], BLOCK_SIZE=64, CBLOCK=32)
        
        # Launch backward inter-block kernel
        _bwd_inter_kernel[grid](GradOut, Q, K, V, GradK, GradV, GradOut.stride(0), Q.stride(0), K.stride(0), V.stride(0), GradK.stride(0), GradV.stride(0), Q.shape[1], Q.shape[2], BLOCK_SIZE=64)
        
        return GradQ, GradK, GradV
