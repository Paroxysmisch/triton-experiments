import triton
import triton.language as tl
import torch

# Heuristics for tile size
def heuristics_for_tile_size(max_tile_size, input_size):
    tile_size = min(max_tile_size, input_size)
    return tile_size

# Heuristics for number of warps
def heuristics_for_num_warps(tile_size):
    if tile_size <= 128:
        return 4
    elif tile_size <= 256:
        return 8
    else:
        return 16

# StridedBuffer class for custom strides
class StridedBuffer:
    def __init__(self, ptr, shape, strides):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides

# Triton kernel for ReLU forward
@triton.jit
def relu_forward_kernel_rank_1(
    X_ptr,  # Pointer to input tensor
    Y_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensor
    stride_x,  # Stride for input tensor
    stride_y,  # Stride for output tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X_ptr + offsets * stride_x, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(Y_ptr + offsets * stride_y, y, mask=mask)

# Wrapper function for ReLU forward
def relu_forward_wrapper_rank_1(X: torch.Tensor, Y: torch.Tensor):
    assert X.dim() == 1, "Input tensor must be 1D"
    assert Y.dim() == 1, "Output tensor must be 1D"
    assert X.shape == Y.shape, "Input and output tensors must have the same shape"

    n_elements = X.numel()
    block_size = heuristics_for_tile_size(1024, n_elements)
    num_warps = heuristics_for_num_warps(block_size)

    grid = (triton.cdiv(n_elements, block_size),)
    relu_forward_kernel_rank_1[grid](
        X, Y, n_elements, X.stride(0), Y.stride(0), BLOCK_SIZE=block_size, num_warps=num_warps
    )

# Example usage
if __name__ == "__main__":
    # Create a 1D tensor
    X = torch.randn(1024, device='cuda')
    Y = torch.empty_like(X)

    # Apply ReLU
    relu_forward_wrapper_rank_1(X, Y)

    # Print the results
    print("Input tensor (X):", X)
    print("Output tensor (Y):", Y)
