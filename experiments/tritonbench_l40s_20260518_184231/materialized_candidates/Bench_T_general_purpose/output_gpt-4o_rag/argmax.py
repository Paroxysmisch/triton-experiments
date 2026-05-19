import torch
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program id which determines which block of the input this instance is processing
    block_id = tl.program_id(0)
    
    # Calculate the starting position of the block
    start = block_id * BLOCK_SIZE
    
    # Load the block into SRAM
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find the maximum value and its index within the block
    max_value = tl.max(values, axis=0)
    max_index = tl.argmax(values, axis=0)
    
    # Write the result to the output
    tl.store(output_ptr + block_id, max_index, mask=mask)

def argmax(input, dim=None, keepdim=False):
    if dim is None:
        # Flatten the input tensor
        input = input.flatten()
        n_elements = input.numel()
        # Allocate output tensor
        output = torch.empty(1, dtype=torch.long, device=input.device)
        # Determine block size
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # Launch kernel
        argmax_kernel[(1,)](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        # Return the index of the maximum value
        return output.item()
    else:
        # Permute dimensions to bring the specified dim to the last
        input = input.transpose(dim, -1)
        shape = input.shape
        n_elements = shape[-1]
        # Allocate output tensor
        output_shape = list(shape)
        output_shape[-1] = 1 if keepdim else 0
        output = torch.empty(*output_shape, dtype=torch.long, device=input.device)
        # Determine block size
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        # Launch kernel for each block
        for idx in range(input.numel() // n_elements):
            argmax_kernel[(1,)](input[idx * n_elements:(idx + 1) * n_elements], output[idx], n_elements, BLOCK_SIZE=BLOCK_SIZE)
        # Reshape the output tensor
        if not keepdim:
            output = output.squeeze(-1)
        return output

# Example usage
input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], dtype=torch.float32, device='cuda')
result = argmax(input_tensor, dim=1, keepdim=True)
print(result)  # Should print the indices of the maximum values along the specified dimension
