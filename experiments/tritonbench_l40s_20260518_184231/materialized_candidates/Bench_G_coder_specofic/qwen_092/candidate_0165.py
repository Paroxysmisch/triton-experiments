import triton
import triton.language as tl

# Define the softmax kernel
@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    row_end = min(row + BLOCK_SIZE, n_cols)

    # Load the input row into SRAM
    input_block = tl.load(input_ptr + row * row_stride, mask=tl.arange(BLOCK_SIZE) < row_end)
    
    # Subtract the max for numerical stability
    max_val = tl.max(input_block)
    input_block -= max_val
    
    # Optionally add the mask
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + row * row_stride, mask=tl.arange(BLOCK_SIZE) < row_end)
        input_block += mask
    
    # Compute the exponentials
    exp_block = tl.exp(input_block)
    
    # Sum the exponentials to derive the denominator
    exp_sum = tl.sum(exp_block, axis=0)
    
    # Divide each element to produce the softmax output
    softmax_block = exp_block / exp_sum
    
    # Store the result back in the output tensor
    tl.store(output_ptr + row * row_stride, softmax_block, mask=tl.arange(BLOCK_SIZE) < row_end)

# Define the wrapper function
@triton.jit
def softmax(input_ptr, output_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE
    row_end = min(row + BLOCK_SIZE, n_cols)

    # Load the input row into SRAM
    input_block = tl.load(input_ptr + row * row_stride, mask=tl.arange(BLOCK_SIZE) < row_end)
    
    # Subtract the max for numerical stability
    max_val = tl.max(input_block)
    input_block -= max_val
    
    # Optionally add the mask
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + row * row_stride, mask=tl.arange(BLOCK_SIZE) < row_end)
        input_block += mask
    
    # Compute the exponentials
    exp_block = tl.exp(input_block)
    
    # Sum the exponentials to derive the denominator
    exp_sum = tl.sum(exp_block, axis=0)
    
    # Divide each element to produce the softmax output
    softmax_block = exp_block / exp_sum
    
    # Store the result back in the output tensor
    tl.store(output_ptr + row * row_stride, softmax_block, mask=tl.arange(BLOCK_SIZE) < row_end)

# Define the Python wrapper function
def softmax(input_tensor, mask_tensor=None, dim=-1):
    if dim != -1:
        raise ValueError("softmax only supports last dimension")
    
    input_tensor = input_tensor.contiguous()
    if mask_tensor is not None:
        mask_tensor = mask_tensor.contiguous()
    
    n_rows, n_cols = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)
    
    # Set up grid and block sizes
    BLOCK_SIZE = 32
    grid = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Invoke the Triton kernel
    softmax_kernel[grid, BLOCK_SIZE](output_tensor.data_ptr(), input_tensor.data_ptr(), input_tensor.stride(0), n_cols, mask_tensor.data_ptr() if mask_tensor is not None else None, BLOCK_SIZE)
    
    return output_tensor
