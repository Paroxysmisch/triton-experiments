import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_features, n_out_features, stride_x, stride_w, stride_b, stride_out,
    p, training, inplace):
    # Calculate the global index
    pid = tl.program_id(axis=0)
    block_size = tl.block_dim(axis=0)
    row_start = pid * block_size
    row_end = min(row_start + block_size, n_out_features)

    # Load data from input tensors
    x_block = tl.load(x_ptr + row_start * stride_x, mask=(row_end - row_start > 0), other=0.)
    
    # Compute linear transformation
    y_block = tl.zeros((block_size,), dtype=x_block.dtype)
    for k in range(n_features):
        y_block += x_block[k] * tl.load(w_ptr + k * stride_w)
    
    # Add bias if it exists
    if b_ptr is not None:
        y_block += tl.load(b_ptr + row_start, mask=(row_end - row_start > 0), other=0.)

    # Sigmoid activation
    y_block = 1 / (1 + tl.exp(-y_block))

    # Dropout
    if training:
        drop_mask = tl.random.rand(block_size) >= p
        y_block *= drop_mask

    # Store the result
    tl.store(out_ptr + row_start * stride_out, y_block, mask=(row_end - row_start > 0))

# Wrapper function
def dropout_sigmoid_linear(input, weight, bias=None, p=0.5, training=True, inplace=False):
    # Get tensor shapes
    n_features = input.shape[-1]
    n_out_features = weight.shape[0]

    # Create output tensor
    if inplace:
        output = input
    else:
        output = input.new_empty((n_out_features, input.shape[0]))

    # Set strides
    stride_x = input.stride(0)
    stride_w = weight.stride(1)
    stride_b = bias.stride(0) if bias is not None else 0
    stride_out = output.stride(0)

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_out_features, meta['BLOCK_SIZE']),)
    dropout_sigmoid_linear_kernel[grid](input.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None, 
                                        output.data_ptr(), n_features, n_out_features, stride_x, stride_w, stride_b, stride_out, 
                                        p, training, inplace)

    return output
