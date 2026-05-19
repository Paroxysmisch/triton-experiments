import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    stride_input,
    stride_other,
    stride_output,
    block_size: int = 256
):
    pid = triton.program_id(0)
    block_start = pid * block_size
    offsets = block_start + triton.arange(0, block_size)

    # Load input and other values
    input_val = input_ptr[offsets]
    other_val = other_ptr[offsets]

    # Compute the result
    result = input_val + alpha * other_val

    # Reduce the results to compute the mean
    total_sum = triton.block_reduce(result, num_splits=n_elements // block_size)

    # Only one thread will have the final sum
    if offsets[0] == 0:
        output_ptr[block_start] = total_sum / n_elements

# Define the grid size and block size
grid_size = lambda n_elements: (n_elements + 255) // 256
block_size = 256

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if isinstance(other, (int, float, complex)):
        other = torch.tensor([other], dtype=input.dtype, device=input.device)

    if dtype is not None:
        input = input.to(dtype)

    if dim is None:
        input_flat = input.flatten()
        other_flat = other.expand_as(input_flat)
        n_elements = input_flat.numel()
        output = torch.zeros_like(input_flat, dtype=torch.float32)
        add_mean_kernel[grid_size(n_elements), block_size](input_flat.data_ptr(), 
                                                          other_flat.data_ptr(),
                                                          output.data_ptr(),
                                                          n_elements,
                                                          alpha,
                                                          input.stride(0),
                                                          other.stride(0),
                                                          output.stride(0))
        return output.view(input.shape).mean().to(dtype) if keepdim else output.mean().item()

    else:
        if isinstance(dim, int):
            dim = (dim,)
        
        reduced_shape = list(input.shape)
        for d in dim:
            reduced_shape[d] = 1
        
        expanded_other = other.expand(reduced_shape)
        n_elements = input.numel() // len(dim)
        output = torch.zeros_like(input, dtype=torch.float32)
        add_mean_kernel[grid_size(n_elements), block_size](input.data_ptr(), 
                                                          expanded_other.data_ptr(),
                                                          output.data_ptr(),
                                                          n_elements,
                                                          alpha,
                                                          input.stride(0),
                                                          expanded_other.stride(0),
                                                          output.stride(0))
        if keepdim:
            return output.view(*reduced_shape)
        else:
            return output.sum(dim=dim).view(-1)

# Example usage
input_tensor = torch.randn(4, 4)
other_tensor = torch.randn(4, 4)
result = add_mean(input_tensor, other_tensor, dim=(0,), alpha=2, keepdim=True)
print(result)
