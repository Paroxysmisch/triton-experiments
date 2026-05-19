import triton
import triton.language as tl
import torch

# Triton kernel for fused hstack and division
@triton.jit
def fused_hstack_div_kernel(
    *tensors_ptr,  # Pointers to the input tensors
    divisor_ptr,   # Pointer to the divisor tensor
    out_ptr,       # Pointer to the output tensor
    n_elements,    # Total number of elements in the output tensor
    n_tensors,     # Number of input tensors
    tensor_widths,  # Widths of the input tensors (for hstack)
    divisor_is_scalar: tl.constexpr,  # Flag indicating if the divisor is a scalar
    rounding_mode: tl.constexpr,      # Rounding mode
    BLOCK_SIZE: tl.constexpr          # Block size for Triton kernel execution
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the divisor
    if divisor_is_scalar:
        divisor = tl.load(divisor_ptr)
    else:
        divisor = tl.load(divisor_ptr + offsets, mask=mask)

    # Initialize the output tensor
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load and stack the input tensors
    offset = 0
    for i in range(n_tensors):
        tensor_width = tensor_widths[i]
        tensor_mask = (offsets >= offset) & (offsets < offset + tensor_width) & mask
        tensor_offsets = offsets - offset
        tensor_data = tl.load(tensors_ptr[i] + tensor_offsets, mask=tensor_mask)
        out += tensor_data
        offset += tensor_width

    # Perform element-wise division with the specified rounding mode
    if rounding_mode == 0:  # None
        out = out / divisor
    elif rounding_mode == 1:  # 'trunc'
        out = tl.where(out >= 0, tl.floor(out / divisor), tl.ceil(out / divisor))
    elif rounding_mode == 2:  # 'floor'
        out = tl.floor(out / divisor)

    # Store the result in the output tensor
    tl.store(out_ptr + offsets, out, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    """
    Performs a fused operation combining horizontal stacking (hstack) and element-wise division.
    
    Parameters:
    - tensors (sequence of Tensors): Sequence of tensors to be horizontally stacked.
    - divisor (Tensor or Number): The tensor or number to divide the stacked tensor by.
    - rounding_mode (str, optional): Type of rounding applied to the result:
      - `None`: Default behavior. Performs no rounding and, if both `input` and `divisor` are integer types, promotes the inputs to the default scalar type. Equivalent to true division in Python (`/` operator).
      - `'trunc'`: Rounds the results of the division towards zero.
      - `'floor'`: Rounds the results of the division down.
      Default: `None`.
    - out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
    
    Returns:
    - Tensor: The resulting tensor after performing the fused operation.
    """
    # Validate input tensors
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("All elements in 'tensors' must be tensors.")
    
    # Validate divisor
    if not isinstance(divisor, (torch.Tensor, int, float)):
        raise ValueError("Divisor must be a tensor or a number.")
    
    # Determine the shape of the stacked tensor
    stacked_shape = list(tensors[0].shape)
    stacked_shape[1] = sum(t.shape[1] for t in tensors)
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(stacked_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    else:
        if out.shape != stacked_shape:
            raise ValueError("Output tensor shape does not match the expected shape.")
    
    # Determine the number of elements in the output tensor
    n_elements = out.numel()
    
    # Determine the block size for Triton kernel execution
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Prepare the tensor widths for hstack
    tensor_widths = [t.shape[1] for t in tensors]
    
    # Prepare the divisor
    if isinstance(divisor, (int, float)):
        divisor = torch.tensor(divisor, dtype=tensors[0].dtype, device=tensors[0].device)
        divisor_is_scalar = True
    else:
        divisor_is_scalar = False
    
    # Determine the rounding mode
    rounding_mode_map = {None: 0, 'trunc': 1, 'floor': 2}
    rounding_mode = rounding_mode_map.get(rounding_mode, 0)
    
    # Prepare the pointers for the Triton kernel
    tensors_ptr = [t.data_ptr() for t in tensors]
    divisor_ptr = divisor.data_ptr()
    out_ptr = out.data_ptr()
    
    # Launch the Triton kernel
    fused_hstack_div_kernel[grid_size, block_size](
        *tensors_ptr,
        divisor_ptr,
        out_ptr,
        n_elements,
        len(tensors),
        tensor_widths,
        divisor_is_scalar,
        rounding_mode,
        BLOCK_SIZE=block_size
    )
    
    return out
