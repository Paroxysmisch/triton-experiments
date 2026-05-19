import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X_ptr,  # pointer to input tensor
    W_ptr,  # pointer to weight tensor 
    Y_ptr,  # pointer to output tensor
    stride,  # stride between rows
    N,      # number of columns
    BLOCK_SIZE: tl.constexpr,  # number of elements to process per block
):
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers for current row
    row_start_ptr = X_ptr + row_idx * stride
    
    # Initialize accumulator for sum of squares
    sum_squares = 0.0
    
    # Load and process elements in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        # Create block mask
        block_mask = block_start + tl.arange(0, BLOCK_SIZE) < N
        
        # Load input elements
        x = tl.load(row_start_ptr + block_start, mask=block_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_squares += tl.sum(x * x * block_mask)
    
    # Compute RMS normalization factor
    rms = tl.sqrt(sum_squares / N + 1e-6)  # add eps for numerical stability
    
    # Normalize and apply weights
    for block_start in range(0, N, BLOCK_SIZE):
        block_mask = block_start + tl.arange(0, BLOCK_SIZE) < N
        
        # Load input and weights
        x = tl.load(row_start_ptr + block_start, mask=block_mask, other=0.0)
        w = tl.load(W_ptr + block_start, mask=block_mask, other=0.0)
        
        # Normalize and multiply by weights
        y = (x / rms) * w
        
        # Store result
        tl.store(Y_ptr + row_idx * stride + block_start, y, mask=block_mask)

class RmsNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Initialize output tensor
        y = torch.empty_like(x)
        
        # Get tensor dimensions
        batch_size, seq_len = x.shape
        
        # Launch kernel
        BLOCK_SIZE = 128
        grid = (batch_size,)
        
        rms_norm_kernel[grid](
            x.data_ptr(),
            self.weight.data_ptr(),
            y.data_ptr(),
            seq_len,  # stride
            seq_len,  # N
            BLOCK_SIZE,
        )
        
        return y

    @staticmethod
    def forward_backward(ctx, x, weight):
        # Save tensors for backward pass
        ctx.save_for_backward(x, weight)
        return RmsNorm.apply_forward(x, weight)
