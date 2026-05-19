import triton
import triton.language as tl

@triton.jit
def relu_kernel(X, Y, stride_xn, stride_xc, stride_xh, stride_xw, stride_yn, stride_yc, stride_yh, stride_yw, N, C, H, W):
    pid = tl.program_id(axis=0)
    n = pid // (C * H * W)
    c = (pid % (C * H * W)) // (H * W)
    h = (pid % (H * W)) // W
    w = pid % W

    # Compute the index in the input tensor
    x_idx = n * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw
    # Apply ReLU
    y_val = tl.max(X[x_idx], 0.0)
    # Compute the index in the output tensor
    y_idx = n * stride_yn + c * stride_yc + h * stride_yh + w * stride_yw
    # Write the result to the output tensor
    Y[y_idx] = y_val

@triton.jit
def fractional_max_pool2d_kernel(X, Y, indices, stride_xn, stride_xc, stride_xh, stride_xw, stride_yn, stride_yc, stride_yh, stride_yw, N, C, H, W, kernel_h, kernel_w, output_h, output_w, return_indices: tl.constexpr):
    pid = tl.program_id(axis=0)
    n = pid // (C * output_h * output_w)
    c = (pid % (C * output_h * output_w)) // (output_h * output_w)
    oh = (pid % (output_h * output_w)) // output_w
    ow = pid % output_w

    # Compute the input region for this output element
    ih_start = int(oh * (H / output_h))
    iw_start = int(ow * (W / output_w))
    ih_end = int((oh + 1) * (H / output_h))
    iw_end = int((ow + 1) * (W / output_w))

    # Initialize max value and index
    max_val = -float('inf')
    max_idx = -1

    for ih in range(ih_start, ih_end):
        for iw in range(iw_start, iw_end):
            # Compute the index in the input tensor
            x_idx = n * stride_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
            # Update max value and index
            if X[x_idx] > max_val:
                max_val = X[x_idx]
                max_idx = x_idx

    # Compute the index in the output tensor
    y_idx = n * stride_yn + c * stride_yc + oh * stride_yh + ow * stride_yw
    # Write the result to the output tensor
    Y[y_idx] = max_val

    if return_indices:
        # Compute the index in the indices tensor
        idx_idx = n * stride_yn + c * stride_yc + oh * stride_yh + ow * stride_yw
        # Write the index of the max value to the indices tensor
        indices[idx_idx] = max_idx

import torch
import triton
import triton.language as tl

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Ensure the input is a 4D tensor (N, C, H, W)
    assert input.dim() == 4, "Input tensor must be 4D (N, C, H, W)"
    
    N, C, H, W = input.shape

    # Determine the output size
    if output_size is not None:
        output_h, output_w = output_size
    elif output_ratio is not None:
        output_h = int(H * output_ratio[0])
        output_w = int(W * output_ratio[1])
    else:
        raise ValueError("Either output_size or output_ratio must be specified")

    # Create output tensor
    output = torch.empty((N, C, output_h, output_w), device=input.device, dtype=input.dtype)

    # Create indices tensor if return_indices is True
    indices = None
    if return_indices:
        indices = torch.empty((N, C, output_h, output_w), device=input.device, dtype=torch.int64)

    # Launch the ReLU kernel
    relu_kernel[(N * C * H * W,)](input, output, input.stride(0), input.stride(1), input.stride(2), input.stride(3), output.stride(0), output.stride(1), output.stride(2), output.stride(3), N, C, H, W)

    # Launch the fractional max pooling kernel
    fractional_max_pool2d_kernel[(N * C * output_h * output_w,)](output, output, indices, output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(0), output.stride(1), output.stride(2), output.stride(3), N, C, H, W, kernel_size[0], kernel_size[1], output_h, output_w, return_indices)

    if return_indices:
        return output, indices
    else:
        return output

# Sample input tensor
input_tensor = torch.randn(2, 3, 10, 10, device='cuda')

# Test the function
output_tensor = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size=(2, 2), output_size=(5, 5))
print(output_tensor)

# Test with indices
output_tensor, indices = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size=(2, 2), output_size=(5, 5), return_indices=True)
print(output_tensor)
print(indices)
