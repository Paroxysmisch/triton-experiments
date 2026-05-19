import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    # Pointers to matrices
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    # Matrix dimensions
    B, N, M, P,
    # Strides for the batch dimension
    stride_b1, stride_b2, stride_bo, stride_bout,
    # Other parameters
    eps, dropout_p,
    # Random seed for dropout
    seed,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute batch index
    bid = tl.program_id(0)
    b_idx = bid // (N // BLOCK_SIZE_M)
    
    # Compute matrix multiplication indices
    pid = tl.program_id(1)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
    
    # Initialize pointers to current batch
    input1_ptr = input1_ptr + b_idx * stride_b1
    input2_ptr = input2_ptr + b_idx * stride_b2
    other_ptr = other_ptr + b_idx * stride_bo
    output_ptr = output_ptr + b_idx * stride_bout
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Batch matrix multiplication
    for k in range(0, M, BLOCK_SIZE_K):
        # Load input blocks
        a = tl.load(input1_ptr + k)
        b = tl.load(input2_ptr + k)
        acc += tl.dot(a, b)
    
    # RMS Normalization
    square_sum = tl.sum(acc * acc, axis=1) / P
    rms = tl.sqrt(square_sum + eps)
    acc = acc / rms[:, None]
    
    # GELU activation
    # Using approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    x = acc
    x3 = x * x * x
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x3)
    acc = x * 0.5 * (1.0 + tl.tanh(inner))
    
    # Dropout
    if dropout_p > 0.0:
        rand = tl.rand(seed, acc.shape)
        mask = rand > dropout_p
        acc = tl.where(mask, acc / (1.0 - dropout_p), 0.0)
    
    # Load and subtract other tensor
    other = tl.load(other_ptr)
    acc = acc - other
    
    # Store result
    tl.store(output_ptr, acc)

class FusedBMMRMSNormGELUDropoutSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, other, normalized_shape, dropout_p=0.5, 
                training=True, approximate='none', eps=1e-5):
        B, N, M = input1.shape
        _, M, P = input2.shape
        
        # Validate shapes
        assert input1.shape[0] == input2.shape[0], "Batch sizes must match"
        assert input1.shape[2] == input2.shape[1], "Inner dimensions must match"
        
        # Initialize output tensor
        output = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
        
        # Configure grid and block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 16
        
        grid = (triton.cdiv(N, BLOCK_SIZE_M) * B,
                triton.cdiv(P, BLOCK_SIZE_N))
        
        # Generate random seed for dropout
        seed = torch.randint(0, 2**31-1, (1,), device=input1.device).item()
        
        # Launch kernel
        fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
            input1, input2, other, output,
            B, N, M, P,
            input1.stride(0), input2.stride(0), 
            other.stride(0), output.stride(0),
            eps, dropout_p if training else 0.0,
            seed,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass if needed
        raise NotImplementedError("Backward pass not implemented yet")

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, 
                                      dropout_p=0.5, training=True, 
                                      approximate='none', eps=1e-5, *, out=None):
    """
    Performs a fused operation combining batch matrix multiplication, RMS normalization,
    GELU activation, dropout, and subtraction.
    
    Args:
        input1 (Tensor): First input tensor (B, N, M)
        input2 (Tensor): Second input tensor (B, M, P)
        other (Tensor): Tensor to subtract
        normalized_shape (int or list or torch.Size): Shape for RMS normalization
        dropout_p (float, optional): Dropout probability. Default: 0.5
        training (bool, optional): Apply dropout if True. Default: True
        approximate (str, optional): GELU approximation ('none' or 'tanh'). Default: 'none'
        eps (float, optional): RMS normalization epsilon. Default: 1e-5
        out (Tensor, optional): Output tensor. Default: None
    
    Returns:
        Tensor: Output tensor of shape (B, N, P)
    """
    return FusedBMMRMSNormGELUDropoutSub.apply(
        input1, input2, other, normalized_shape, dropout_p, 
        training, approximate, eps
    )
