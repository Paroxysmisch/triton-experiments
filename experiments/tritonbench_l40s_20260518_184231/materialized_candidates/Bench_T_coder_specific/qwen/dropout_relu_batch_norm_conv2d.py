import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    norm_mean_ptr,
    norm_var_ptr,
    scale_ptr,
    offset_ptr,
    stride,
    padding,
    dilation,
    groups,
    n, c_in, h_out, w_out, c_out, kh, kw,
    p,
    training,
    inplace,
    block_size: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(n, block_size)
    
    row = pid % grid_n
    col = pid // grid_n
    
    # Load input, weight, bias
    x = tl.load(input_ptr + row * h_out * w_out * c_in + col * c_in * h_out * w_out)
    w = tl.load(weight_ptr + (col // groups) * c_out * kh * kw + ((row % (kh * kw)) // kh) * c_out + (row % kh))
    b = tl.load(bias_ptr + col * c_out)
    
    # Perform convolution
    acc = tl.zeros((c_out,), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            idx = (row // (kh * kw)) * h_out * w_out + ((row % (kh * kw)) // kh) * w_out + (row % kh)
            acc += x[idx] * w[i * kw + j]
    
    # Add bias
    acc += b
    
    # Batch normalization
    mean = tl.mean(acc, axis=0)
    var = tl.var(acc, axis=0)
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    normalized_acc = (acc - mean) * inv_std
    
    # ReLU activation
    relu_acc = tl.maximum(normalized_acc, 0.0)
    
    # Dropout
    if training:
        mask = tl.random.rand() > p
        relu_acc *= mask
    
    # Store result
    tl.store(output_ptr + row * h_out * w_out * c_out + col * c_out * h_out * w_out, relu_acc)
