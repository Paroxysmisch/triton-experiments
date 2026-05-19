import triton
import triton.language as tl
import torch

# Triton kernel for checking finiteness of elements in a rank-1 tensor
@triton.jit
def isfinite_func_kernel_rank_1(
    in0_ptr,  # Pointer to the input tensor
    out_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    one_tile_per_cta: tl.constexpr,  # Flag to determine if one tile per CTA
    BLOCK_SIZE: tl.constexpr  # Block size for the grid-stride loop
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data elements from the input tensor
    in0 = tl.load(in0_ptr + offsets, mask=mask)

    # Apply the isfinite function
    out = tl.where(tl.math.isfinite(in0), 1, 0)

    # Store the results back into the output tensor
    tl.store(out_ptr + offsets, out, mask=mask)

# Wrapper function for the isfinite kernel
def isfinite_func_wrapper_rank_1(in0, out, one_tile_per_cta=True):
    # Ensure input and output tensors have matching shapes
    assert in0.shape == out.shape, "Input and output tensors must have the same shape"

    # Determine optimal tile sizes and number of warps
    def heuristics_for_tile_size(n_elements):
        if n_elements < 1024:
            return 128
        else:
            return 256

    def heuristics_for_num_warps(n_elements):
        if n_elements < 1024:
            return 4
        else:
            return 8

    n_elements = in0.numel()
    BLOCK_SIZE = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(n_elements)

    # Calculate task parameters
    num_ctas = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_ctas,)

    # Launch the Triton kernel
    isfinite_func_kernel_rank_1[grid](
        in0, out, n_elements, one_tile_per_cta, BLOCK_SIZE, num_warps=num_warps
    )

# Example usage
if __name__ == "__main__":
    # Create input and output tensors
    in0 = torch.tensor([1.0, 2.0, float('inf'), float('nan'), 3.0], device='cuda')
    out = torch.zeros_like(in0, device='cuda')

    # Call the wrapper function
    isfinite_func_wrapper_rank_1(in0, out)

    # Print the results
    print("Input tensor:", in0)
    print("Output tensor (finiteness check):", out)
