import triton
import triton.language as tl
import torch

@triton.jit
def _pow_kernel(
    input_ptr,      # pointer to input data
    exponent_ptr,   # pointer to exponent data (when not scalar)
    output_ptr,     # pointer to output data
    exponent_scalar, # scalar exponent value
    is_exponent_scalar, # bool flag indicating scalar exponent
    numel,          # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_offset = program_id * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load exponent value (scalar or tensor)
    exp_val = tl.where(
        is_exponent_scalar,
        exponent_scalar,
        tl.load(exponent_ptr + offsets, mask=mask, other=0.0)
    )

    # Compute x^exp_val using tl.pow
    out_val = tl.pow(x, exp_val)
    
    # Store result
    tl.store(output_ptr + offsets, out_val, mask=mask)

def pow(input, exponent, *, out=None):
    # Handle device placement and type
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor.")

    # Determine broadcast if exponent is a tensor
    if isinstance(exponent, (int, float)):
        exponent_scalar = float(exponent)
        is_exponent_scalar = True
        expanded_exponent = None
    else:
        if not exponent.is_cuda:
            raise ValueError("exponent tensor must be a CUDA tensor.")
        # Broadcast shapes
        broadcast_shape = torch.broadcast_shapes(input.shape, exponent.shape)
        # Expand both to the broadcast shape
        input = input.expand(broadcast_shape)
        exponent = exponent.expand(broadcast_shape)
        exponent_scalar = 0.0
        is_exponent_scalar = False
        expanded_exponent = exponent
    
    # If necessary, broadcast input to match
    if not isinstance(exponent, (int, float)):
        input = input.expand(broadcast_shape)

    # Allocate output if not provided
    if out is None:
        if isinstance(exponent, (int, float)):
            out = torch.empty_like(input)
        else:
            out = torch.empty_like(input, shape=broadcast_shape)
    else:
        # Ensure out is on CUDA
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor.")
        # Broadcast out if needed
        if isinstance(exponent, (int, float)):
            out.resize_(input.shape)
        else:
            out.resize_(broadcast_shape)

    # Flatten tensors for simple 1D kernel launch
    input_flat = input.flatten()
    out_flat = out.flatten()
    
    if expanded_exponent is not None:
        exp_flat = expanded_exponent.flatten()
    else:
        exp_flat = None

    numel = input_flat.numel()

    # Define block size
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (numel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )

    # Launch Triton kernel
    _pow_kernel[grid](
        input_flat.data_ptr(),
        0 if exp_flat is None else exp_flat.data_ptr(),
        out_flat.data_ptr(),
        exponent_scalar,
        1 if is_exponent_scalar else 0,
        numel,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.reshape(input.shape if is_exponent_scalar else broadcast_shape)
