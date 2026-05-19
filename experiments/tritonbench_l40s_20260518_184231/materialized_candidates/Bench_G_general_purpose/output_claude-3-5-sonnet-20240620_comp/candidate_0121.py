import triton
import triton.language as tl
import torch

@triton.jit
def relu_forward_kernel_rank_1(
    output_ptr, input_ptr,
    stride_out, stride_in,
    size, BLOCK_SIZE: tl.constexpr
):
    # Calculate the program ID
    pid = tl.program_id(0)
    
    # Calculate the block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < size
    
    # Load input data using the mask
    x = tl.load(input_ptr + offsets * stride_in, mask=mask)
    
    # Compute ReLU: max(0, x)
    output = tl.where(x > 0, x, 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets * stride_out, output, mask=mask)

def heuristics_for_tile_size(size):
    # Maximum tile size allowed
    MAX_TILE_SIZE = 2048
    
    # Calculate optimal tile size
    tile_size = min(size, MAX_TILE_SIZE)
    
    # Round down to nearest power of 2 for better performance
    tile_size = 2 ** int.bit_length(tile_size - 1)
    return tile_size

def heuristics_for_num_warps(tile_size):
    # Determine number of warps based on tile size
    if tile_size <= 256:
        return 4
    elif tile_size <= 512:
        return 8
    else:
        return 16

class StridedBuffer:
    def __init__(self, ptr, stride):
        self.ptr = ptr
        self.stride = stride

def relu_forward_wrapper_rank_1(input_tensor):
    # Get input properties
    size = input_tensor.numel()
    device = input_tensor.device
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate optimal tile size and number of warps
    BLOCK_SIZE = heuristics_for_tile_size(size)
    num_warps = heuristics_for_num_warps(BLOCK_SIZE)
    
    # Calculate grid size
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    # Get strides
    input_stride = input_tensor.stride(0)
    output_stride = output.stride(0)
    
    # Launch kernel
    relu_forward_kernel_rank_1[grid](
        output.data_ptr(),
        input_tensor.data_ptr(),
        output_stride,
        input_stride,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output

# Example usage
def test_relu():
    # Create test input
    input_tensor = torch.randn(1000000, device='cuda')
    
    # Run Triton ReLU
    output = relu_forward_wrapper_rank_1(input_tensor)
    
    # Verify against PyTorch's ReLU
    torch_output = torch.nn.functional.relu(input_tensor)
    assert torch.allclose(output, torch_output)
    print("ReLU test passed!")
