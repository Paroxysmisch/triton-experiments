import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_shape, weight_shape, bias_ptr, stride, padding, dilation, groups, upscale_factor,
    BLOCK_SIZE=32):
    """
    input_ptr: Pointer to the input tensor
    weight_ptr: Pointer to the weight tensor
    output_ptr: Pointer to the output tensor
    input_shape: Shape of the input tensor (minibatch, in_channels, iH, iW)
    weight_shape: Shape of the weight tensor (out_channels, in_channels/groups, kH, kW)
    bias_ptr: Pointer to the bias tensor (out_channels), can be None
    stride: Stride of the convolving kernel
    padding: Padding added to all four sides of the input
    dilation: Spacing between kernel elements
    groups: Number of blocked connections from input channels to output channels
    upscale_factor: Factor by which to increase spatial resolution
    BLOCK_SIZE: Block size for parallelism
    """
    # Extract shape information
    N, C_in, H_in, W_in = input_shape
    C_out, C_in_per_group, K_h, K_w = weight_shape
    assert C_out % upscale_factor**2 == 0, "Output channels must be divisible by upscale factor squared"
    C_out_per_group = C_out // upscale_factor**2
    
    # Calculate output shape
    H_out = H_in * upscale_factor
    W_out = W_in * upscale_factor
    
    # Get thread indices
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    
    # Compute base index for input and weight
    n_base = n * C_out_per_group + c_out // upscale_factor**2
    c_in_base = c_out % upscale_factor**2
    
    # Loop over spatial positions
    h_in = tl.arange(0, H_in)
    w_in = tl.arange(0, W_in)
    h_out = h_in * upscale_factor
    w_out = w_in * upscale_factor
    
    # Initialize output value
    out_val = 0.0
    
    # Load input and weight data
    input_val = tl.load(input_ptr + n_base * C_in * H_in * W_in + h_in * W_in + w_in, mask=(h_in < H_in) & (w_in < W_in))
    weight_val = tl.load(weight_ptr + c_in_base * K_h * K_w + h_out * K_w + w_out, mask=(c_in_base < C_in_per_group) & (h_out < K_h) & (w_out < K_w))
    
    # Accumulate the result
    out_val += input_val * weight_val
    
    # Write the result to output
    tl.store(output_ptr + n * C_out * H_out * W_out + h_out * W_out + w_out, out_val)

# Wrapper function
def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Validate input shapes
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape
    assert C_out % upscale_factor**2 == 0, "Output channels must be divisible by upscale factor squared"
    C_out_per_group = C_out // upscale_factor**2
    
    # Allocate output tensor
    H_out = H_in * upscale_factor
    W_out = W_in * upscale_factor
    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Launch Triton kernel
    grid = lambda meta: (
        triton.cdiv(N * C_out_per_group, meta['BLOCK_SIZE']),
        triton.cdiv(H_in * W_in, meta['BLOCK_SIZE'])
    )
    pixel_shuffle_conv2d_kernel[grid](input.data_ptr(), weight.data_ptr(), output.data_ptr(),
                                      input.shape, weight.shape, bias.data_ptr() if bias is not None else None,
                                      stride, padding, dilation, groups, upscale_factor,
                                      BLOCK_SIZE=32)
    
    return output
