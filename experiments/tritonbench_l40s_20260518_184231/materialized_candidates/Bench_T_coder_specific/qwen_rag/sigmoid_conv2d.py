import triton
import triton.language as tl

@triton.jit
def conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, N, C, H, W, K, KH, KW, BLOCK_SIZE: tl.constexpr
):
    # Thread index within the block
    n = tl.program_id(0)
    c = tl.program_id(1)
    kh = tl.program_id(2)
    kw = tl.program_id(3)
    
    # Compute spatial indices
    h = n * stride_h + kh * dilation_h - padding_h
    w = c * stride_w + kw * dilation_w - padding_w
    
    # Initialize the sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load input, weight, and bias
    for ci in range(groups):
        input_offset = n * C * H * W + ci * (H * W) + (h * W + w)
        weight_offset = ci * (K * KH * KW) + (kh * KW + kw)
        
        # Bounds check for input
        if h >= 0 and h < H and w >= 0 and w < W:
            input_val = tl.load(input_ptr + input_offset)
        else:
            input_val = 0.0
        
        # Bounds check for weight
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate the product
        acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_offset = c
        bias_val = tl.load(bias_ptr + bias_offset)
        acc += bias_val
    
    # Sigmoid activation
    output = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store the result
    output_ptr[n * K * H * W + c * (H * W) + (kh * W + kw)] = output
