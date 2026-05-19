import triton
import triton.language as tl

@triton.jit
def fused_tile_exp_kernel(output_ptr, input_ptr, input_shape, output_shape, strides_in, strides_out, num_repetitions, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Calculate the index within the input tensor
    indices = [row * strides_in[i] + col * strides_in[i+1] for i in range(len(strides_in)-1)]
    input_val = tl.load(input_ptr + sum(indices))
    
    # Repeat the input value according to the specified number of repetitions
    replicated_vals = [input_val for _ in range(num_repetitions)]
    
    # Apply the exponential function element-wise
    exp_vals = [tl.exp(val) for val in replicated_vals]
    
    # Store the result in the output tensor
    output_indices = [row * strides_out[i] + col * strides_out[i+1] for i in range(len(strides_out)-1)]
    tl.store(output_ptr + sum(output_indices), sum(exp_vals))

def fused_tile_exp(input, dims, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    input_shape = input.shape
    output_shape = tuple(input_shape[i] * dim for i, dim in enumerate(dims))
    
    # Determine the strides for the input and output tensors
    strides_in = list(itertools.accumulate([1] + list(reversed(input_shape))))
    strides_out = list(itertools.accumulate([1] + list(reversed(output_shape))))
    
    # Pad dims with ones if necessary
    dims = dims + (1,) * (len(input_shape) - len(dims))
    
    BLOCK_SIZE = 128
    
    fused_tile_exp_kernel[grid=tuple(math.ceil(dim/BLOCK_SIZE) for dim in output_shape), 
                          block=(BLOCK_SIZE, 1, 1)](out.data_ptr(), 
                                                  input.data_ptr(), 
                                                  input_shape, 
                                                  output_shape, 
                                                  strides_in, 
                                                  strides_out, 
                                                  dims)
    
    return out
