import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(X_ptr, W_ptr, Y_ptr, D1, D2, N, stride_x1, stride_x2, stride_x3, stride_w2, stride_w3, BLOCK_SIZE: tl.constexpr):
    # Block index
    pid = tl.program_id(0)
    
    # Compute the position of the block in the grid
    d1 = pid // D2
    d2 = pid % D2
    
    # Offsets for this block
    x_offset = d1 * stride_x1 + d2 * stride_x2
    w_offset = d2 * stride_w2
    
    # Create a pointer to the start of this block in X and W
    X_block_ptr = X_ptr + x_offset
    W_block_ptr = W_ptr + w_offset
    
    # Create pointers for output
    Y_block_ptr = Y_ptr + x_offset
    
    # Load the input and weights for this block
    X = tl.load(X_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    W = tl.load(W_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N)
    
    # Compute mean
    mean = tl.sum(X, axis=0) / N
    
    # Compute variance
    var = tl.sum((X - mean) ** 2, axis=0) / N
    
    # Normalize
    X_hat = (X - mean) / tl.sqrt(var + 1e-5)
    
    # Scale with weights
    Y = X_hat * W
    
    # Store the result
    tl.store(Y_block_ptr + tl.arange(0, BLOCK_SIZE), Y, mask=tl.arange(0, BLOCK_SIZE) < N)

def layernorm_forward(X, W, BLOCK_SIZE=128):
    D1, D2, N = X.shape
    assert W.shape == (D2, N), "Weight shape must match the last two dimensions of X"
    
    # Compute strides
    stride_x1, stride_x2, stride_x3 = X.stride()
    stride_w2, stride_w3 = W.stride()
    
    # Prepare output tensor
    Y = torch.empty_like(X)
    
    # Compute grid size
    grid = (D1 * D2,)
    
    # Launch the kernel
    _layer_norm_fwd_kernel[grid](X, W, Y, D1, D2, N, stride_x1, stride_x2, stride_x3, stride_w2, stride_w3, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y
