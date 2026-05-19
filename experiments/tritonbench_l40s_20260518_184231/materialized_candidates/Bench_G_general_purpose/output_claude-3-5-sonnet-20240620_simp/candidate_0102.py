import torch
import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for the different dimensions
    stride_b, stride_h, stride_s,
    # Scale for attention
    scale,
    # Decay factor
    decay,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head index
    batch_id = pid // num_heads
    head_id = pid % num_heads
    
    # Compute starting position
    start_q = batch_id * stride_b + head_id * stride_h
    start_k = batch_id * stride_b + head_id * stride_h
    start_v = batch_id * stride_b + head_id * stride_h
    
    # Initialize the output
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load decay powers
    decay_powers = tl.power(decay, tl.arange(0, BLOCK_SIZE))
    
    # Loop over sequence length in blocks
    for i in range(0, seq_len, BLOCK_SIZE):
        # Load query block
        q = tl.load(q_ptr + start_q + i * stride_s,
                   mask=i < seq_len, other=0.0)
        
        # Initialize accumulator for this block
        block_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Inner loop for key-value pairs
        for j in range(0, i + BLOCK_SIZE):
            # Load key and value
            k = tl.load(k_ptr + start_k + j * stride_s,
                       mask=j < seq_len, other=0.0)
            v = tl.load(v_ptr + start_v + j * stride_s,
                       mask=j < seq_len, other=0.0)
            
            # Compute attention scores
            scores = q * k * scale
            
            # Apply decay factor based on position difference
            scores = scores * decay_powers[i - j]
            
            # Accumulate weighted values
            block_acc += scores * v
            
        # Add block result to main accumulator
        acc += block_acc
        
    # Store the result
    tl.store(out_ptr + start_q + tl.arange(0, BLOCK_SIZE) * stride_s,
             acc, mask=tl.arange(0, BLOCK_SIZE) < seq_len)

@triton.jit
def parallel_retention_bwd_kernel(
    # Pointers to matrices
    dout_ptr, q_ptr, k_ptr, v_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, head_dim,
    # Strides for the different dimensions
    stride_b, stride_h, stride_s,
    # Scale and decay factors
    scale, decay,
    BLOCK_SIZE: tl.constexpr
):
    # Similar structure to forward kernel
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads
    
    # Compute starting positions
    start_pos = batch_id * stride_b + head_id * stride_h
    
    # Initialize gradients
    dq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dv = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load decay powers
    decay_powers = tl.power(decay, tl.arange(0, BLOCK_SIZE))
    
    # Main backward pass loop
    for i in range(0, seq_len, BLOCK_SIZE):
        # Load gradients of output
        grad_out = tl.load(dout_ptr + start_pos + i * stride_s,
                          mask=i < seq_len, other=0.0)
        
        # Load corresponding q, k, v
        q = tl.load(q_ptr + start_pos + i * stride_s,
                   mask=i < seq_len, other=0.0)
        k = tl.load(k_ptr + start_pos + i * stride_s,
                   mask=i < seq_len, other=0.0)
        v = tl.load(v_ptr + start_pos + i * stride_s,
                   mask=i < seq_len, other=0.0)
        
        # Compute gradients
        dq += grad_out * k * v * scale * decay_powers
        dk += grad_out * q * v * scale * decay_powers
        dv += grad_out * q * k * scale * decay_powers
        
    # Store gradients
    tl.store(dq_ptr + start_pos + tl.arange(0, BLOCK_SIZE) * stride_s,
             dq, mask=tl.arange(0, BLOCK_SIZE) < seq_len)
    tl.store(dk_ptr + start_pos + tl.arange(0, BLOCK_SIZE) * stride_s,
             dk, mask=tl.arange(0, BLOCK_SIZE) < seq_len)
    tl.store(dv_ptr + start_pos + tl.arange(0, BLOCK_SIZE) * stride_s,
             dv, mask=tl.arange(0, BLOCK_SIZE) < seq_len)

class ParallelRetention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=1.0, decay=0.9):
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.decay = decay
        
        # Extract dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Compute strides
        stride_b = seq_len * num_heads * head_dim
        stride_h = seq_len * head_dim
        stride_s = head_dim
        
        # Allocate output
        output = torch.empty_like(q)
        
        # Launch kernel
        grid = (batch_size * num_heads,)
        parallel_retention_fwd_kernel[grid](
            q, k, v, output,
            batch_size, seq_len, num_heads, head_dim,
            stride_b, stride_h, stride_s,
            scale, decay,
            BLOCK_SIZE=32
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        decay = ctx.decay
        
        # Extract dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Compute strides
        stride_b = seq_len * num_heads * head_dim
        stride_h = seq_len * head_dim
        stride_s = head_dim
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch backward kernel
        grid = (batch_size * num_heads,)
        parallel_retention_bwd_kernel[grid](
            grad_output, q, k, v,
            dq, dk, dv,
            batch_size, seq_len, num_heads, head_dim,
            stride_b, stride_h, stride_s,
            scale, decay,
            BLOCK_SIZE=32
        )
        
        return dq, dk, dv, None, None

def parallel_retention(q, k, v, scale=1.0, decay=0.9):
    return ParallelRetention.apply(q, k, v, scale, decay)
