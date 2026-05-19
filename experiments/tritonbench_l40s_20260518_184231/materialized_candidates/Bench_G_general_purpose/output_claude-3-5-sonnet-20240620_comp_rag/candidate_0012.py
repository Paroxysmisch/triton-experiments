import triton
import triton.language as tl
import torch
import math

@triton.jit
def rms_matmul_rbe(
    # Pointers to matrices
    x_ptr, w_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_dim, output_dim,
    # Strides
    stride_xb, stride_xs, stride_xh,
    stride_wb, stride_wh, stride_wo,
    stride_ob, stride_os, stride_oh,
    # RMS norm params
    eps: tl.float32,
    # Optional rotary embedding params
    use_rbe: tl.int32,
    pos_offset: tl.int32,
    theta: tl.float32,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_dim, BLOCK_SIZE_N)
    num_pid_b = batch_size
    
    # Block ID
    bid_b = pid // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    bid_m = pid_mn // num_pid_n
    bid_n = pid_mn % num_pid_n
    
    # Block pointers
    x_block_ptr = x_ptr + bid_b * stride_xb + bid_m * BLOCK_SIZE_M * stride_xs
    w_block_ptr = w_ptr + bid_b * stride_wb + bid_n * BLOCK_SIZE_N * stride_wo
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load RMS weights
    rms_w = tl.load(rms_w_ptr + tl.arange(0, BLOCK_SIZE_K))
    
    # Iterate over k dimension
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # Load x block and apply RMS norm
        x = tl.load(x_block_ptr + k + tl.arange(0, BLOCK_SIZE_K))
        x_squared = x * x
        rms = tl.sqrt(tl.sum(x_squared) / hidden_dim + eps)
        x_normalized = x / rms
        x_scaled = x_normalized * rms_w
        
        # Apply rotary embeddings if enabled
        if use_rbe:
            pos = bid_m * BLOCK_SIZE_M + pos_offset
            freq = 1.0 / (theta ** (tl.arange(0, BLOCK_SIZE_K, 2) / hidden_dim))
            cos = tl.cos(pos * freq)
            sin = tl.sin(pos * freq)
            x_rot_even = x_scaled[::2] * cos - x_scaled[1::2] * sin
            x_rot_odd = x_scaled[::2] * sin + x_scaled[1::2] * cos
            x_scaled = tl.concatenate([x_rot_even, x_rot_odd])
        
        # Load weights block
        w = tl.load(w_block_ptr + k * stride_wh + tl.arange(0, BLOCK_SIZE_K))
        
        # Matrix multiply
        acc += tl.dot(x_scaled, w)
    
    # Store output
    output_block_ptr = output_ptr + bid_b * stride_ob + bid_m * stride_os + bid_n * stride_oh
    tl.store(output_block_ptr, acc)

class RMSMatMulRBE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, rms_weight, use_rbe=False, pos_offset=0, theta=10000.0):
        batch_size, seq_len, hidden_dim = x.shape
        output_dim = weight.shape[-1]
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = weight.contiguous()
        rms_weight = rms_weight.contiguous()
        
        # Compute block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = min(32, triton.next_power_of_2(hidden_dim))
        
        # Allocate output
        output = torch.empty((batch_size, seq_len, output_dim), 
                           device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (batch_size * triton.cdiv(seq_len, BLOCK_SIZE_M) * 
                triton.cdiv(output_dim, BLOCK_SIZE_N),)
        
        rms_matmul_rbe[grid](
            x, weight, rms_weight, output,
            batch_size, seq_len, hidden_dim, output_dim,
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(0), weight.stride(1), weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            eps=1e-6,
            use_rbe=use_rbe,
            pos_offset=pos_offset,
            theta=theta,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return output

def rms_matmul_rbe_wrapper(x, weight, rms_weight, use_rbe=False, pos_offset=0, theta=10000.0):
    """
    Wrapper function for RMS normalized matrix multiplication with optional rotary embeddings
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        weight: Weight matrix of shape (hidden_dim, output_dim)
        rms_weight: RMS normalization weights of shape (hidden_dim,)
        use_rbe: Whether to apply rotary embeddings
        pos_offset: Position offset for rotary embeddings
        theta: Base for rotary embedding frequency computation
    
    Returns:
        Output tensor of shape (batch_size, seq_len, output_dim)
    """
    return RMSMatMulRBE.apply(x, weight, rms_weight, use_rbe, pos_offset, theta)
