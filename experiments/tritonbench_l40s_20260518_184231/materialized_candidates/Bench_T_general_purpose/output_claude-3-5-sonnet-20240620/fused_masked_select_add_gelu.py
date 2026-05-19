# ... existing imports ...
import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(input_ptr, mask_ptr, other_ptr, output_ptr, alpha, N):
    # Get the index for the current thread
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    
    # Ensure we are within bounds
    mask = tl.load(mask_ptr + idx)
    input = tl.load(input_ptr + idx)
    other = tl.load(other_ptr + idx)

    # Perform masked selection
    selected = tl.where(mask, input, 0.0)  # Replace with 0.0 if mask is False
    # Add alpha-scaled other tensor
    result = selected + alpha * other
    
    # Apply GELU activation
    output = 0.5 * result * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (result + 0.044715 * result ** 3)))
    
    # Store the result
    tl.store(output_ptr + idx, output)

def fused_masked_select_add_gelu(input: torch.Tensor, mask: torch.Tensor, other: torch.Tensor, *, alpha: float = 1, approximate: str = 'none', out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Validate input shapes
    if input.shape != mask.shape or input.shape != other.shape:
        raise ValueError("Input, mask, and other must have the same shape.")
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements
    N = input.numel()
    
    # Launch the Triton kernel
    grid = (N + 255) // 256  # Assuming a block size of 256
    fused_masked_select_add_gelu_kernel[grid](input, mask, other, out, alpha, N)
    
    return out
