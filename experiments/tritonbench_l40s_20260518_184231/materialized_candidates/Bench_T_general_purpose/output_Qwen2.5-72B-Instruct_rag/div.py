import triton
import triton.language as tl
import torch

# Helper function to determine the block size
def get_block_size(n_elements):
    return triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))

# Helper function to determine the grid size
def get_grid_size(n_elements, block_size):
    return triton.cdiv(n_elements, block_size)

# Kernel function for when both 'input' and 'other' are tensors.
@triton.jit
def div_tensor_tensor_kernel(input_ptr, other_ptr, out_ptr, n_elements, block_size: tl.constexpr, rounding_mode: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    
    if rounding_mode == 0:  # 'true' division
        out_val = input_val / other_val
    elif rounding_mode == 1:  # 'floor' division
        out_val = tl.floor(input_val / other_val)
    elif rounding_mode == 2:  # 'trunc' division
        out_val = tl.trunc(input_val / other_val)
    
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Kernel function for when 'input' is a tensor and 'other' is a scalar.
@triton.jit
def div_tensor_scalar_kernel(input_ptr, other, out_ptr, n_elements, block_size: tl.constexpr, rounding_mode: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    
    if rounding_mode == 0:  # 'true' division
        out_val = input_val / other
    elif rounding_mode == 1:  # 'floor' division
        out_val = tl.floor(input_val / other)
    elif rounding_mode == 2:  # 'trunc' division
        out_val = tl.trunc(input_val / other)
    
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Kernel function for when 'input' is a scalar and 'other' is a tensor.
@triton.jit
def div_scalar_tensor_kernel(input, other_ptr, out_ptr, n_elements, block_size: tl.constexpr, rounding_mode: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    other_val = tl.load(other_ptr + offsets, mask=mask)
    
    if rounding_mode == 0:  # 'true' division
        out_val = input / other_val
    elif rounding_mode == 1:  # 'floor' division
        out_val = tl.floor(input / other_val)
    elif rounding_mode == 2:  # 'trunc' division
        out_val = tl.trunc(input / other_val)
    
    tl.store(out_ptr + offsets, out_val, mask=mask)

# Wrapper function to select and call the appropriate kernel based on the input types.
def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = get_block_size(n_elements)
    grid_size = get_grid_size(n_elements, block_size)
    
    rounding_mode_map = {
        None: 0,
        'floor': 1,
        'trunc': 2
    }
    
    rounding_mode = rounding_mode_map.get(rounding_mode, 0)
    
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        div_tensor_tensor_kernel[(grid_size, 1, 1)](input, other, out, n_elements, block_size, rounding_mode)
    elif isinstance(input, torch.Tensor):
        div_tensor_scalar_kernel[(grid_size, 1, 1)](input, other, out, n_elements, block_size, rounding_mode)
    elif isinstance(other, torch.Tensor):
        div_scalar_tensor_kernel[(grid_size, 1, 1)](input, other, out, n_elements, block_size, rounding_mode)
    else:
        raise ValueError("At least one of 'input' or 'other' must be a tensor.")
    
    return out
