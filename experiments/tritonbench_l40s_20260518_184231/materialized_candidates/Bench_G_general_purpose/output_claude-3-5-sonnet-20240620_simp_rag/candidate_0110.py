import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(
    Input, Weight, Bias, Output, Mean, Var,
    stride, N,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused layer normalization kernel that computes:
    y = ((x - mean) / sqrt(var + eps)) * weight + bias
    
    Parameters:
        Input: input tensor (S, D)
        Weight: gamma parameter
        Bias: beta parameter
        Output: output tensor
        Mean: mean of each row
        Var: variance of each row
        stride: stride of input tensor
        N: size of last dimension
        EPS: epsilon for numerical stability
        BLOCK_SIZE: size of thread block
    """
    # Get the row index
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute input pointers and load data
    input_ptr = Input + row * stride + cols
    x = tl.load(input_ptr, mask=cols < N, other=0.0).to(tl.float32)
    
    # Load weight and bias
    weight = tl.load(Weight + cols, mask=cols < N, other=0.0).to(tl.float32)
    bias = tl.load(Bias + cols, mask=cols < N, other=0.0).to(tl.float32)
    
    # Compute mean
    row_mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_centered = x - row_mean
    row_var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Store mean and variance
    tl.store(Mean + row, row_mean)
    tl.store(Var + row, row_var)
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(row_var + EPS)
    x_norm = x_centered * inv_std
    
    # Apply weight and bias
    output = x_norm * weight + bias
    
    # Store result
    output_ptr = Output + row * stride + cols
    tl.store(output_ptr, output, mask=cols < N)

def fused_native_layer_norm(primals_1, primals_2, primals_3):
    """
    Wrapper function for the layer normalization kernel.
    
    Args:
        primals_1: weight (gamma) tensor
        primals_2: bias (beta) tensor
        primals_3: input tensor of shape (S, D)
    
    Returns:
        tuple: (normalized tensor, input tensor, mean, auxiliary output)
    """
    # Get input dimensions
    S, D = primals_3.shape
    
    # Initialize output buffers
    buf4 = torch.empty_like(primals_3)  # normalized output
    buf0 = torch.empty(S, dtype=primals_3.dtype, device=primals_3.device)  # mean
    buf3 = torch.empty(S, dtype=primals_3.dtype, device=primals_3.device)  # variance
    
    # Compute maximum block size
    MAX_FUSED_SIZE = 65536 // primals_3.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_SIZE:
        raise RuntimeError("Feature dimension too large (>= 64KB)")
    
    # Launch kernel
    grid = (S,)
    triton_red_fused_native_layer_norm_0[grid](
        primals_3, primals_1, primals_2, buf4, buf0, buf3,
        primals_3.stride(0), D,
        EPS=1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return buf4, primals_3, buf0, buf3
