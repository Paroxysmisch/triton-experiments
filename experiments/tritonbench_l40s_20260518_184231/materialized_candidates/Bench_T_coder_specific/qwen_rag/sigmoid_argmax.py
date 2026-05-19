import triton
import triton.language as tl

@triton.jit
def sigmoid_argmax_kernel(output_ptr, input_ptr, input_row_stride, input_col_stride, output_row_stride, output_col_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    grid_size = tl.cdiv(n_rows * n_cols, BLOCK_SIZE)
    row, col = pid % n_rows, pid // n_rows
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_data = tl.load(input_ptr + row * input_row_stride + col * input_col_stride + offsets, mask=offsets < n_cols, other=-float('inf'))
    
    # Sigmoid computation
    sigmoid_values = 1 / (1 + tl.exp(-input_data))
    
    # Argmax computation
    max_val = tl.max(sigmoid_values)
    max_indices = tl.where(sigmoid_values == max_val, offsets, -1)
    
    # Reduce across the block
    max_indices_reduced = tl.all_reduce(max_indices, op=tl.RedOp.SUM, num_splits=num_stages)
    
    # Store the result
    output_ptr[pid] = max_indices_reduced

def sigmoid_argmax(input, dim=None, keepdim=False):
    if dim is None:
        # Flatten the input tensor
        input = input.view(-1)
        n_elements = input.size(0)
        output = torch.zeros((1,), dtype=torch.long, device=input.device)
        
        sigmoid_argmax_kernel[1, 1](output, input, 0, 1, 0, 1, n_elements, 1, BLOCK_SIZE=128, num_stages=4)
        
        if not keepdim:
            return output.view()
        else:
            return output.view(1)
    else:
        input_shape = input.shape
        n_rows, n_cols = input_shape[dim], input_shape[dim + 1]
        
        output_shape = list(input_shape)
        if not keepdim:
            output_shape.pop(dim)
        
        output = torch.zeros(output_shape, dtype=torch.long, device=input.device)
        
        sigmoid_argmax_kernel[input_shape[0], input_shape[1]](output, input, input.stride(0), input.stride(1), output.stride(0), output.stride(1), n_rows, n_cols, BLOCK_SIZE=128, num_stages=4)
        
        return output
