import triton
import torch

@triton.jit
def adaptive_avg_pool2d_kernel(
    X,
    Y,
    stride_H,
    stride_W,
    padding_H,
    padding_W,
    grid=(1, 1, 1),
):
    """
    X : [N, C, H_in, W_in]
    Y : [N, C, H_out, W_out]
    """
    n = tl.program_id(0)
    c = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Compute the start indices for the input tensor
    h_start = h_out * stride_H - padding_H
    w_start = w_out * stride_W - padding_W

    # Compute the end indices for the input tensor
    h_end = min(h_start + stride_H, X.shape[2])
    w_end = min(w_start + stride_W, X.shape[3])

    # Initialize the sum and count for averaging
    sum_val = tl.zeros([], dtype=tl.float32)
    count = tl.zeros([], dtype=tl.int32)

    # Iterate over the input region
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            sum_val += X[n, c, h, w]
            count += 1

    # Compute the average and write it to the output tensor
    avg_val = sum_val / count
    Y[n, c, h_out, w_out] = avg_val

def adaptive_avg_pool2d(input_tensor, output_size):
    N, C, H_in, W_in = input_tensor.shape
    
    if isinstance(output_size, int):
        H_out = W_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("Invalid output_size format")
    
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in
    
    # Create output tensor
    output_tensor = torch.empty((N, C, H_out, W_out), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Calculate strides and padding
    stride_H = H_in // H_out
    stride_W = W_in // W_out
    padding_H = (H_out * stride_H - H_in) // 2
    padding_W = (W_out * stride_W - W_in) // 2
    
    # Launch Triton kernel
    block = (8, 8, 1)
    grid = ((N * C * H_out + block[0] - 1) // block[0], 
            (W_out + block[1] - 1) // block[1], 
            1)
    
    adaptive_avg_pool2d_kernel[grid](input_tensor, output_tensor, stride_H, stride_W, padding_H, padding_W, block=block)
    
    return output_tensor
