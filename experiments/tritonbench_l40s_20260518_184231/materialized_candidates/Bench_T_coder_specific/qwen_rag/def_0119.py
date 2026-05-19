import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_conv2d_kernel(
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr, 
    input_shape, 
    weight_shape, 
    bias_shape, 
    stride, 
    padding, 
    dilation, 
    groups, 
    upscale_factor, 
    BLOCK_SIZE: tl.constexpr
):
    # Unpack shapes
    N, C_in, H_in, W_in = input_shape
    C_out, _, kH, kW = weight_shape
    
    # Program ID
    n = tl.program_id(0)
    c = tl.program_id(1)
    
    # Block offsets
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    
    # Thread offsets within block
    h_thread = tl.arange(0, BLOCK_SIZE)
    w_thread = tl.arange(0, BLOCK_SIZE)
    
    # Compute input indices
    h_in = h_block * BLOCK_SIZE + h_thread
    w_in = w_block * BLOCK_SIZE + w_thread
    
    # Compute output indices
    h_out = h_in * stride - padding + kH // 2
    w_out = w_in * stride - padding + kW // 2
    
    # Initialize output value
    out_value = 0.0
    
    # Loop over filters and weights
    for f in range(C_out):
        for g in range(groups):
            for kh in range(kH):
                for kw in range(kW):
                    in_ch = g * (C_in // groups) + (f % (C_in // groups))
                    in_idx = n * C_in * H_in * W_in + in_ch * H_in * W_in + \
                             (h_in + kh - kH // 2) * W_in + (w_in + kw - kW // 2)
                    weight_idx = f * C_in * kH * kW + in_ch * kH * kW + kh * kW + kw
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Convolution accumulation
                    out_value += tl.load(input_ptr + in_idx) * weight_val
                    
    # Apply bias
    if bias_ptr is not None:
        bias_idx = f
        bias_val = tl.load(bias_ptr + bias_idx)
        out_value += bias_val
    
    # Check bounds for pixel shuffle
    if (h_out >= 0 and h_out < N * upscale_factor and
        w_out >= 0 and w_out < C_out * (H_in * W_in)):
        
        # Pixel shuffle rearrangement
        new_h = h_out // upscale_factor
        new_w = (h_out % upscale_factor) * upscale_factor + w_out
        
        # Store output
        out_idx = n * C_out * H_in * W_in + new_h * W_in + new_w
        tl.store(output_ptr + out_idx, out_value)
