import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_no_welford(
    X,  # input tensor
    Y,  # output tensor
    Mean,  # mean tensor
    Rstd,  # inverse standard deviation tensor
    weight,  # weight tensor
    bias,  # bias tensor
    stride,  # stride of the input tensor
    N,  # number of elements in the reduction dimension
    eps,  # small value to avoid division by zero
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the start and end indices for the block
    block_start = pid * XBLOCK
    block_end = min(block_start + XBLOCK, stride)
    
    # Initialize the mean and variance accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Iterate over the block
    for i in range(block_start, block_end):
        # Load the input value
        x = tl.load(X + i * N)
        # Accumulate the mean and variance
        sum_x += x
        sum_x2 += x * x
    
    # Compute the mean and variance for the block
    mean = sum_x / N
    var = (sum_x2 / N) - (mean * mean)
    
    # Compute the inverse standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store the mean and inverse standard deviation
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    
    # Normalize the input tensor
    for i in range(block_start, block_end):
        # Load the input value
        x = tl.load(X + i * N)
        # Normalize the input value
        y = (x - mean) * rstd
        # Apply weight and bias
        y = y * tl.load(weight + i) + tl.load(bias + i)
        # Store the normalized value
        tl.store(Y + i * N, y)

import torch
import triton
import triton.language as tl

def fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3, eps=1e-5):
    # Ensure inputs are on the same device
    device = primals_1.device
    assert primals_2.device == device and primals_3.device == device
    
    # Get the shape of the input tensor
    N = primals_1.shape[-1]
    stride = primals_1.numel() // N
    
    # Allocate output tensors
    Y = torch.empty_like(primals_1)
    Mean = torch.empty((stride,), device=device, dtype=torch.float32)
    Rstd = torch.empty((stride,), device=device, dtype=torch.float32)
    
    # Define block sizes
    XBLOCK = 128
    RBLOCK = 128
    
    # Launch the kernel
    grid = (stride // XBLOCK + (stride % XBLOCK > 0),)
    triton_red_fused_native_layer_norm_no_welford[grid](
        primals_1, Y, Mean, Rstd, primals_2, primals_3, stride, N, eps, XBLOCK, RBLOCK
    )
    
    return Y, Mean, Rstd
