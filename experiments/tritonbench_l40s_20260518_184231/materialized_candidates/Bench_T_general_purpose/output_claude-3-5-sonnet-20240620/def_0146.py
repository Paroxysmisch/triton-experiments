import torch
import triton
import triton.language as tl

@triton.jit
def elu_linear_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Parameters
    alpha,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Optional bias flag
    HAS_BIAS: tl.constexpr,
):
    """
    Computes ELU(Linear(input)) in a fused operation
    """
    pid = tl.program_id(axis=0)
    
    # Block index
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointers for current block
    input_block_ptr = input_ptr + block_m * BLOCK_SIZE_M * K
    weight_block_ptr = weight_ptr + block_n * BLOCK_SIZE_N
    
    # Iterate through K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load input and weight blocks
        input_block = tl.load(input_block_ptr + k)
        weight_block = tl.load(weight_block_ptr + k * N)
        
        # Compute matrix multiplication
        acc += tl.dot(input_block, weight_block)
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + block_n * BLOCK_SIZE_N)
        acc += bias
    
    # Apply ELU activation
    # ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    mask = acc <= 0
    acc = tl.where(mask, alpha * (tl.exp(acc) - 1), acc)
    
    # Store result
    output_block_ptr = output_ptr + block_m * BLOCK_SIZE_M * N + block_n * BLOCK_SIZE_N
    tl.store(output_block_ptr, acc)

def elu_linear(input: torch.Tensor, 
               weight: torch.Tensor, 
               bias: torch.Tensor = None, 
               alpha: float = 1.0, 
               inplace: bool = False) -> torch.Tensor:
    """
    Applies linear transformation followed by ELU activation.
    
    Args:
        input (Tensor): Input tensor of shape (batch_size, in_features)
        weight (Tensor): Weight matrix of shape (out_features, in_features)
        bias (Tensor, optional): Bias vector of shape (out_features)
        alpha (float, optional): Alpha parameter for ELU. Default: 1.0
        inplace (bool, optional): Whether to perform the operation in-place. Default: False
    
    Returns:
        Tensor: Output tensor of shape (batch_size, out_features)
    """
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert weight.dim() == 2, "Weight must be a 2D tensor"
    
    batch_size, in_features = input.shape
    out_features, weight_in_features = weight.shape
    
    assert in_features == weight_in_features, "Input features must match weight dimensions"
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), 
                        device=input.device, 
                        dtype=input.dtype)
    
    # Grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = (batch_size * out_features + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Launch kernel
    elu_linear_kernel[grid](
        input_ptr=input, 
        weight_ptr=weight,
        output_ptr=output,
        bias_ptr=bias if bias is not None else None,
        M=batch_size, 
        N=out_features, 
        K=in_features,
        alpha=alpha,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
    )
    
    return output
