import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['T']
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,  # Input tensor pointer
    o,  # Output tensor pointer
    s_s_h: tl.constexpr,  # Stride for the head dimension (H)
    s_s_t: tl.constexpr,  # Stride for the sequence dimension (T)
    T: tl.constexpr,       # Total sequence length
    BT: tl.constexpr,      # Block size for T (tunable parameter)
):
    i_bh = tl.program_id(0)  # Batch and head index combined
    
    # Create an upper triangular mask for reversed cumulative sum
    o_i = tl.arange(0, BT)
    mask = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    
    # Initialize accumulation variable
    b_z = 0.0
    
    # Iterate over T in reverse order with block size BT
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        start_idx = i_t * BT
        
        # Define block pointer for current input block
        p_s = tl.make_block_ptr(
            base=s + i_bh * s_s_h,
            shape=(T,),
            strides=(s_s_t,),
            offsets=(start_idx,),
            block_shape=(BT,),
            order=(0,)
        )
        # Load the current block of input
        s_block = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        
        # Compute reversed cumulative sum within the block using matrix multiplication
        reversed_cumsum_block = tl.dot(s_block[None, :], mask)
        reversed_cumsum_block = reversed_cumsum_block[0, :]  # Flatten to 1D
        
        # Add the accumulated sum from previous blocks
        output_block = reversed_cumsum_block + b_z
        
        # Define block pointer for output
        p_o = tl.make_block_ptr(
            base=o + i_bh * s_s_h,
            shape=(T,),
            strides=(s_s_t,),
            offsets=(start_idx,),
            block_shape=(BT,),
            order=(0,)
        )
        # Store the computed output block
        tl.store(p_o, output_block.to(p_o.dtype.element_ty), boundary_check=(0,))
        
        # Update the accumulated sum with the current block's total
        sum_block = tl.sum(s_block)
        b_z += sum_block


def chunk_global_reversed_cumsum_scalar(s: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the reversed cumulative sum kernel.
    
    Args:
        s (torch.Tensor): Input tensor of shape (B, H, T)
        
    Returns:
        torch.Tensor: Output tensor of the same shape as input with reversed cumulative sums
    """
    B, H, T = s.shape
    o = torch.empty_like(s)
    
    # Define grid size (B*H)
    grid = (B * H, )
    
    # Launch the kernel with autotuned configurations
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2),
        T=T
    )
    
    return o
