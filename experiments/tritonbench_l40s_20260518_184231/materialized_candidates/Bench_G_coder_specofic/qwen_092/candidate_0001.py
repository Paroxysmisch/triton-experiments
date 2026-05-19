import triton
import torch

def dequantize_rowwise(x, state_x, inv_127, n_elements, BLOCK_SIZE=32):
    """
    Wrapper function for row-wise dequantization of the input tensor 'x'.
    """
    # Prepare the output tensor
    output = torch.empty_like(x)
    
    # Compute P2 for efficient memory alignment
    P2 = 2 ** (int(triton.next_power_of_two(x.shape[1])) - 1)
    
    # Set up the execution grid
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    
    # Invoke the Triton kernel
    _dequantize_rowwise[grid](x, state_x, output, inv_127, n_elements, BLOCK_SIZE, P2, tl.program_id(0))
    
    return output
