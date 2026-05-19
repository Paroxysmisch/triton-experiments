import triton
import triton.language as tl
import torch

@triton.jit
def _fused_mv_logsoftmax_dropout_kernel(
    input_ptr, vec_ptr, output_ptr,
    n_rows, n_cols,
    p, seed,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < n_rows
    
    # Load vector
    vec = tl.load(vec_ptr + tl.arange(0, n_cols))
    
    # Initialize accumulator for matrix-vector multiplication
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load row of input matrix and perform matrix-vector multiplication
    row_offs = row_idx[:, None] * n_cols + tl.arange(0, n_cols)[None, :]
    row = tl.load(input_ptr + row_offs, mask=mask[:, None], other=0.0)
    acc = tl.sum(row * vec, axis=1)
    
    # Apply log-softmax
    # First compute max for numerical stability
    row_max = tl.max(acc, axis=0)
    acc = acc - row_max
    exp_x = tl.exp(acc)
    sum_exp = tl.sum(exp_x)
    log_sum_exp = tl.log(sum_exp)
    log_softmax = acc - log_sum_exp
    
    # Apply dropout during training
    if training:
        random = tl.rand(seed, row_idx)
        dropout_mask = random > p
        log_softmax = tl.where(dropout_mask, log_softmax / (1.0 - p), 0.0)
    
    # Store result
    tl.store(output_ptr + row_idx, log_softmax, mask=mask)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    """
    Fused operation combining matrix-vector multiplication, log-softmax, and dropout.
    
    Args:
        input (torch.Tensor): Input matrix of shape (n, m)
        vec (torch.Tensor): Input vector of shape (m,)
        p (float): Dropout probability
        training (bool): Whether to apply dropout
        inplace (bool): Whether to perform operation in-place
        dim (int): Dimension along which to compute log-softmax (0 or -1)
        out (torch.Tensor, optional): Output tensor
        
    Returns:
        torch.Tensor: Result of fused operation
    """
    assert dim in (0, -1), "dim must be 0 or -1"
    assert input.dim() == 2, "input must be 2-dimensional"
    assert vec.dim() == 1, "vec must be 1-dimensional"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    
    # Handle device and dtype
    device = input.device
    dtype = input.dtype
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(input.size(0), device=device, dtype=dtype)
    elif not inplace:
        assert out.size() == (input.size(0),), "out must have shape (n,)"
    
    # Get dimensions
    n_rows, n_cols = input.shape
    
    # Configure grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)
    
    # Generate random seed for dropout
    seed = torch.randint(0, 2**31-1, (1,), device=device).item() if training else 0
    
    # Launch kernel
    _fused_mv_logsoftmax_dropout_kernel[grid](
        input_ptr=input.data_ptr(),
        vec_ptr=vec.data_ptr(),
        output_ptr=out.data_ptr(),
        n_rows=n_rows,
        n_cols=n_cols,
        p=p,
        seed=seed,
        training=training,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
