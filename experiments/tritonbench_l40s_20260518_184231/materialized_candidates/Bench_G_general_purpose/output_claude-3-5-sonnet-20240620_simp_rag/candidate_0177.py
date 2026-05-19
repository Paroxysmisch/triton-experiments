import math
import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_vector_kernel(
    # Pointers to matrices
    input_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, num_heads, chunk_size,
    # Strides for the input and output tensors
    stride_batch, stride_seq, stride_head,
    # Block sizes for optimization
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Compute offsets
    offs_batch = pid_batch * stride_batch
    offs_head = pid_head * stride_head
    
    # Initialize offsets for the sequence dimension
    offs_seq = tl.arange(0, BLOCK_SIZE)
    
    # Compute input/output pointers
    input_block_ptr = input_ptr + offs_batch + offs_head
    output_block_ptr = output_ptr + offs_batch + offs_head
    
    # Load input block
    x = tl.load(input_block_ptr + offs_seq * stride_seq,
                mask=offs_seq < seq_len,
                other=0.0)
    
    # Compute cumulative sum within chunk
    chunk_idx = offs_seq // chunk_size
    chunk_offset = offs_seq % chunk_size
    
    # Create mask for cumsum operation
    mask = tl.arange(0, BLOCK_SIZE) <= chunk_offset
    
    # Perform chunked cumsum
    x_cumsum = tl.where(mask, x, 0.0)
    x_cumsum = tl.cumsum(x_cumsum, axis=0)
    
    # Store result
    tl.store(output_block_ptr + offs_seq * stride_seq,
             x_cumsum,
             mask=offs_seq < seq_len)

# Wrapper function
def chunk_global_cumsum_vector(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Compute chunk-based cumulative sum of a vector.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, num_heads]
        chunk_size: Size of chunks for cumulative sum
    
    Returns:
        Output tensor of same shape with chunk-wise cumulative sum
    """
    batch_size, seq_len, num_heads = x.shape
    
    # Ensure input is contiguous and in correct format
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Configure kernel parameters
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    grid = (batch_size, num_heads)
    
    # Launch kernel
    chunk_global_cumsum_vector_kernel[grid](
        x, output,
        batch_size, seq_len, num_heads, chunk_size,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE,
    )
    
    return output

# Backward pass kernel
@triton.jit
def chunk_global_cumsum_vector_backward_kernel(
    grad_output_ptr, input_ptr, grad_input_ptr,
    batch_size, seq_len, num_heads, chunk_size,
    stride_batch, stride_seq, stride_head,
    BLOCK_SIZE: tl.constexpr,
):
    # Similar structure to forward kernel
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    offs_batch = pid_batch * stride_batch
    offs_head = pid_head * stride_head
    offs_seq = tl.arange(0, BLOCK_SIZE)
    
    # Load gradient
    grad = tl.load(grad_output_ptr + offs_batch + offs_head + offs_seq * stride_seq,
                   mask=offs_seq < seq_len,
                   other=0.0)
    
    # Compute backward pass within chunks
    chunk_idx = offs_seq // chunk_size
    chunk_offset = offs_seq % chunk_size
    
    # Create reverse mask for backward cumsum
    mask = tl.arange(0, BLOCK_SIZE) >= chunk_offset
    
    # Perform chunked backward cumsum
    grad_input = tl.where(mask, grad, 0.0)
    grad_input = tl.cumsum(grad_input, axis=0)
    
    # Store result
    tl.store(grad_input_ptr + offs_batch + offs_head + offs_seq * stride_seq,
             grad_input,
             mask=offs_seq < seq_len)
