import torch
import triton
import triton.language as tl

@triton.jit
def triton_layer_norm_kernel(
    X, Mean, Var, Y, 
    Gamma, Beta,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Compute row index and column indices
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute pointers to input and output
    x_ptr = X + row * stride + cols
    y_ptr = Y + row * stride + cols
    
    # Load input values
    x = tl.load(x_ptr, mask=cols < N, other=0.0).to(tl.float32)
    
    # Compute mean
    x_sum = tl.sum(x, axis=0)
    x_mean = x_sum / N
    
    # Compute variance
    x_zm = x - x_mean
    x_zm = tl.where(cols < N, x_zm, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    
    # Store mean and variance
    tl.store(Mean + row, x_mean)
    tl.store(Var + row, x_var)
    
    # Normalize
    rstd = 1 / tl.sqrt(x_var + eps)
    y = x_zm * rstd
    
    # Apply affine transformation
    gamma = tl.load(Gamma + cols, mask=cols < N, other=1.0)
    beta = tl.load(Beta + cols, mask=cols < N, other=0.0)
    y = y * gamma + beta
    
    # Store output
    tl.store(y_ptr, y, mask=cols < N)

def fused_native_layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    # Reshape input data into 2D tensor
    x_reshaped = x.reshape(-1, x.shape[-1])
    M, N = x_reshaped.shape
    
    # Determine block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    
    # Determine number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    
    # Prepare output tensors
    y = torch.empty_like(x)
    mean = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    var = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    
    # Enqueue kernel
    grid = (M,)
    triton_layer_norm_kernel[grid](
        x_reshaped, mean, var, y.reshape(-1, N),
        gamma, beta,
        x_reshaped.stride(0),
        N, eps,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y, mean, var

# Example usage
batch_size, seq_len, hidden_size = 32, 128, 512
x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
gamma = torch.ones(hidden_size, device='cuda')
beta = torch.zeros(hidden_size, device='cuda')

y, mean, var = fused_native_layer_norm(x, gamma, beta)
print(f"Output shape: {y.shape}")
print(f"Mean shape: {mean.shape}")
print(f"Variance shape: {var.shape}")
