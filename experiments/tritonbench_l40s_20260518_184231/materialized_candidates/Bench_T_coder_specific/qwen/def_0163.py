import triton
import triton.language as tl

@triton.jit
def cos_kernel(x_ptr, cos_out_ptr, n_elements, BLOCK_SIZE=256):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    cos_x = tl.cos(x)
    tl.store(cos_out_ptr + offsets, cos_x)
    return

@triton.jit
def signbit_kernel(cos_x_ptr, signbit_out_ptr, n_elements, BLOCK_SIZE=256):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    cos_x = tl.load(cos_x_ptr + offsets)
    signbit = tl.bitcast(tl.trunc_bits(cos_x, 32, 31), tl.uint8)
    tl.store(signbit_out_ptr + offsets, signbit)
    return

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get the device and dtype
    device = input.device
    dtype = input.dtype
    
    # Ensure the input is a float tensor
    if dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32")
    
    # Allocate output tensors
    cos_output = torch.empty_like(input)
    signbit_output = torch.empty_like(input, dtype=torch.bool)
    
    # Number of elements in the input tensor
    n_elements = input.numel()
    
    # Configure grid and block sizes
    grid_size = (n_elements + 255) // 256
    
    # Launch the cosine kernel
    cos_kernel[grid_size, 256](input.contiguous().ptr(), cos_output.contiguous().ptr(), n_elements)
    
    # Launch the signbit kernel
    signbit_kernel[grid_size, 256](cos_output.contiguous().ptr(), signbit_output.contiguous().ptr(), n_elements)
    
    return cos_output, signbit_output
