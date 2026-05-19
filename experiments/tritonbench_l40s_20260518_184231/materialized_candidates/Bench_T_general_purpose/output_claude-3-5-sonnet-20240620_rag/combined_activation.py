import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    # Pointers to inputs and outputs
    Output, Input, Weight1, Weight2, Bias,
    # Dimensions
    Batch, N, D_in, D_out,
    # Strides for the input and output tensors
    stride_batch, stride_n, stride_din, stride_dout,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid = tl.program_id(0)
    
    # Calculate batch and N indices
    batch_idx = pid // N
    n_idx = pid % N
    
    # Create ranges for reduction
    r_din = tl.arange(0, BLOCK_SIZE)
    
    # Initialize pointers
    input_ptr = Input + batch_idx * stride_batch + n_idx * stride_n
    weight1_ptr = Weight1 + n_idx * D_out
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Matrix multiplication (X @ W1)
    for i in range(0, D_in, BLOCK_SIZE):
        mask = r_din < (D_in - i)
        x = tl.load(input_ptr + r_din * stride_din, mask=mask, other=0.0)
        w1 = tl.load(weight1_ptr + r_din, mask=mask, other=0.0)
        acc += x * w1
    
    # Apply sigmoid
    acc = 1 / (1 + tl.exp(-acc))
    
    # Apply tanh
    exp_pos = tl.exp(acc)
    exp_neg = tl.exp(-acc)
    acc = (exp_pos - exp_neg) / (exp_pos + exp_neg)
    
    # Load weight2 and bias for element-wise operations
    w2 = tl.load(Weight2 + n_idx * D_out)
    b = tl.load(Bias + n_idx * D_out)
    
    # Element-wise multiplication with weight2 and add bias
    acc = acc * w2 + b
    
    # Store result
    output_ptr = Output + batch_idx * stride_batch + n_idx * stride_dout
    tl.store(output_ptr, acc)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    """
    Performs combined activation operation: Y = (tanh(sigmoid(X @ W1)) ⊙ W2) + b
    
    Args:
        input (Tensor): Input tensor of shape (*, N, D_in)
        weight1 (Tensor): Weight matrix of shape (D_in, D_out)
        weight2 (Tensor): Weight tensor for element-wise multiplication
        bias (Tensor): Bias tensor
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: Output tensor of shape (*, N, D_out)
    """
    # Get dimensions
    *batch_dims, N, D_in = input.shape
    D_out = weight1.shape[1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Input validation
    assert weight1.shape == (D_in, D_out), "Weight1 dimensions mismatch"
    assert weight2.shape[-1] == D_out, "Weight2 dimensions mismatch"
    assert bias.shape[-1] == D_out, "Bias dimensions mismatch"
    
    # Ensure contiguous tensors
    input = input.contiguous()
    weight1 = weight1.contiguous()
    weight2 = weight2.contiguous()
    bias = bias.contiguous()
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(*batch_dims, N, D_out, 
                         device=input.device, 
                         dtype=input.dtype)
    
    # Calculate strides
    stride_batch = N * D_in
    stride_n = D_in
    stride_din = 1
    stride_dout = 1
    
    # Configure grid and block sizes
    BLOCK_SIZE = 32
    grid = (batch_size * N,)
    
    # Launch kernel
    combined_activation_kernel[grid](
        out, input, weight1, weight2, bias,
        batch_size, N, D_in, D_out,
        stride_batch, stride_n, stride_din, stride_dout,
        BLOCK_SIZE
    )
    
    return out
