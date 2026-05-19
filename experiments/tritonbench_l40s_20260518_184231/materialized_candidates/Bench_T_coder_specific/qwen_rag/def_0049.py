import triton
import triton.language as tl
import torch
from torch.nn.functional import conv2d

# Define the LeakyReLU kernel
@triton.jit
def leaky_relu_kernel(X, Y, N, C, H, W, OH, OW, stride, padding, dilation, groups, negative_slope):
    pid = tl.program_id(axis=0)
    X_idx = pid // (OH * OW)
    Y_idx = pid % (OH * OW)
    
    # Compute spatial indices
    oy = Y_idx // OW
    ox = Y_idx % OW
    
    # Compute input indices
    iy = oy * stride - padding
    ix = ox * stride - padding
    
    # Initialize accumulation
    y = 0.0
    
    # Convolution loop
    for kh in range(H):
        ky = iy + kh * dilation
        for kw in range(W):
            kx = ix + kw * dilation
            if ky >= 0 and ky < H and kx >= 0 and kx < W:
                ci = X_idx * C + (kh * W + kw) // groups
                y += X[ci * H * W + ky * W + kx]
    
    # Apply LeakyReLU
    y = tl.max(y, 0.0) + negative_slope * tl.min(y, 0.0)
    
    # Store result
    Y[Y_idx] = y

# Define the wrapper function
def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Get input dimensions
    N, C_in, H, W = input.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    
    # Calculate output dimensions
    OH = (H + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    
    # Allocate output tensor
    if inplace:
        output = input
    else:
        output = torch.zeros((N, C_out, OH, OW), device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    num_elements = OH * OW
    grid_size = (num_elements + triton.cdiv(num_elements, 1024)) // 1024
    block_size = 1024
    leaky_relu_kernel[input.shape[0], block_size](input.contiguous(), output, N, C_out, C_in, H, W, OH, OW, stride, padding, dilation, groups, negative_slope)
    
    return output

# Example usage
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 32, 32, device='cuda')
    weight_tensor = torch.randn(64, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(64, device='cuda')
    
    output = leaky_relu_conv2d(input_tensor, weight_tensor, bias=bias_tensor, stride=1, padding=1, dilation=1, groups=1, negative_slope=0.01, inplace=False)
    print(output.shape)
