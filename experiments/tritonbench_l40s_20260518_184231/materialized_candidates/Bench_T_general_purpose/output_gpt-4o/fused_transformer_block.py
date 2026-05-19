import triton
import triton.language as tl
import torch
import torch.nn.functional as F

@triton.jit
def fused_transformer_block_kernel(
    X_ptr, W1_ptr, W2_ptr, R_ptr, out_ptr,
    gamma_ptr, beta_ptr, dropout_mask_ptr,
    N, D_in, D_k, D_out,
    dropout_p, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Pointers to block of data
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Compute Z1 = X @ W1
    X_block = tl.load(X_ptr + row_idx * D_in + col_idx, mask=True)
    W1_block = tl.load(W1_ptr + col_idx * D_k, mask=True)
    Z1 = tl.dot(X_block, W1_block)
    
    # Apply softmax to Z1
    Z1_max = tl.max(Z1, axis=1)
    Z1_exp = tl.exp(Z1 - Z1_max)
    Z2 = Z1_exp / tl.sum(Z1_exp, axis=1)
    
    # Apply dropout to Z2
    dropout_mask = tl.load(dropout_mask_ptr + row_idx * D_k + col_idx, mask=True)
    Z3 = Z2 * dropout_mask * (1.0 / (1.0 - dropout_p))
    
    # Compute Z4 = Z3 @ W2
    W2_block = tl.load(W2_ptr + col_idx * D_out, mask=True)
    Z4 = tl.dot(Z3, W2_block)
    
    # Add residual
    R_block = tl.load(R_ptr + row_idx * D_out + col_idx, mask=True)
    Z4_residual = Z4 + R_block
    
    # Apply LayerNorm
    mean = tl.mean(Z4_residual, axis=1)
    var = tl.var(Z4_residual, axis=1)
    Z4_norm = (Z4_residual - mean) / tl.sqrt(var + eps)
    
    gamma = tl.load(gamma_ptr + col_idx, mask=True)
    beta = tl.load(beta_ptr + col_idx, mask=True)
    Y = gamma * Z4_norm + beta
    
    # Store result
    tl.store(out_ptr + row_idx * D_out + col_idx, Y, mask=True)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    N, D_in = input.shape[-2:]
    D_k = weight1.shape[-1]
    D_out = weight2.shape[-1]
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Generate dropout mask
    dropout_mask = (torch.rand_like(input) > dropout_p).float()
    
    # LayerNorm parameters (gamma and beta)
    gamma = torch.ones(D_out, device=input.device)
    beta = torch.zeros(D_out, device=input.device)
    
    # Launch Triton kernel
    grid = (N, D_out)
    fused_transformer_block_kernel[grid](
        input, weight1, weight2, residual, out,
        gamma, beta, dropout_mask,
        N, D_in, D_k, D_out,
        dropout_p, eps,
        BLOCK_SIZE=128
    )
    
    return out
