import triton
import triton.language as tl

@triton.jit
def fused_transformer_kernel(input_ptr, weight1_ptr, weight2_ptr, residual_ptr, out_ptr, 
                             dropout_p, eps, N, D_in, D_k, D_out):
    # Define the grid size
    batch_idx = tl.program_id(0)
    row_idx = tl.arange(0, D_k)
    
    # Load input and weights
    input_tensor = tl.load(input_ptr + batch_idx * D_in + row_idx)
    weight1 = tl.load(weight1_ptr + row_idx)
    
    # Matrix multiplication: Z1 = X * W1
    Z1 = tl.dot(input_tensor, weight1)
    
    # Softmax: Z2 = softmax(Z1)
    Z2 = tl.softmax(Z1, dim=-1)
    
    # Dropout: Z3 = dropout(Z2)
    Z3 = tl.dropout(Z2, p=dropout_p)
    
    # Load second weight matrix
    weight2 = tl.load(weight2_ptr + row_idx)
    
    # Matrix multiplication: Z4 = Z3 * W2
    Z4 = tl.dot(Z3, weight2)
    
    # Load residual tensor
    residual_tensor = tl.load(residual_ptr + batch_idx * D_out + row_idx)
    
    # Residual connection: Y = LayerNorm(Z4 + R)
    Y = Z4 + residual_tensor
    Y = tl.layer_norm(Y, eps=eps)
    
    # Store the output
    tl.store(out_ptr + batch_idx * D_out + row_idx, Y)

def fused_transformer_block(input: Tensor, weight1: Tensor, weight2: Tensor, 
                            residual: Tensor, dropout_p: float = 0.1, 
                            eps: float = 1e-5, *, out: Optional[Tensor] = None) -> Tensor:
    # Ensure input shapes are compatible
    assert input.shape[-1] == weight1.shape[0], "Input and weight1 dimensions must match."
    
    # Get dimensions
    N, D_in = input.shape[:-1], input.shape[-1]
    D_k = weight1.shape[1]
    D_out = weight2.shape[1]
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(*N, D_out, device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    grid = (N[0],)  # Assuming N is the batch size
    fused_transformer_kernel[grid](input, weight1, weight2, residual, out, 
                                    dropout_p, eps, N[0], D_in, D_k, D_out)
    
    return out
