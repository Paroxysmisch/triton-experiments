import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def sqrt_exp_kernel(
    X_ptr,
    Y_ptr,
    N_elements,
    BLOCK_SIZE: int = 256
):
    pid = triton.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = triton.arange(0, BLOCK_SIZE)
    indices = block_start + offsets

    # Ensure we don't access out-of-bounds elements
    mask = indices < N_elements

    x = X_ptr[indices] * mask
    y = triton.math.sqrt(x)
    y = triton.math.exp(y)

    Y_ptr[indices] = y * mask

# Define the wrapper function
def sqrt_exp(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)

    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
    
    sqrt_exp_kernel[grid](input.data_ptr(), out.data_ptr(), input.numel())

    return out

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float32)
    result = sqrt_exp(input_tensor)
    print(result)
