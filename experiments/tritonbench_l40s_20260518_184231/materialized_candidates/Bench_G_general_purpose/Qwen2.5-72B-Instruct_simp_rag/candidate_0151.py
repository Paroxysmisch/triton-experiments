import torch
import triton
import triton.language as tl

# Triton kernel for the forward pass of the softmax operation
@triton.jit
def _softmax(output_ptr, input_ptr, mask_ptr, n_batch, n_heads, n_cols, BLOCK_SIZE: tl.constexpr,
             log: tl.constexpr, causal: tl.constexpr, mask_type: tl.constexpr):
    pid = tl.program_id(0)
    batch_start = pid // (n_heads * n_cols)
    head_start = (pid % (n_heads * n_cols)) // n_cols
    col_start = (pid % (n_heads * n_cols)) % n_cols

    batch_offset = batch_start * n_heads * n_cols * n_cols
    head_offset = head_start * n_cols * n_cols
    col_offset = col_start * n_cols

    input_base_ptr = input_ptr + batch_offset + head_offset + col_offset
    output_base_ptr = output_ptr + batch_offset + head_offset + col_offset
    mask_base_ptr = mask_ptr + batch_offset + head_offset + col_offset if mask_ptr != 0 else 0

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_base_ptr + col_offsets
    input_row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    if mask_type == 1:  # Apply mask
        mask_row = tl.load(mask_base_ptr + col_offsets, mask=mask, other=-float('inf'))
        input_row = input_row + mask_row

    if causal:  # Apply causal constraint
        causal_mask = col_offsets < col_start
        input_row = tl.where(causal_mask, -float('inf'), input_row)

    input_row_minus_max = input_row - tl.max(input_row, axis=0)
    numerator = tl.exp(input_row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    if log:  # Apply log transformation
        softmax_output = tl.log(softmax_output)

    output_ptrs = output_base_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

# Triton kernel for the backward pass of the softmax operation
@triton.jit
def _softmax_backward(grad_output_ptr, output_ptr, grad_input_ptr, n_batch, n_heads, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    batch_start = pid // (n_heads * n_cols)
    head_start = (pid % (n_heads * n_cols)) // n_cols
    col_start = (pid % (n_heads * n_cols)) % n_cols

    batch_offset = batch_start * n_heads * n_cols * n_cols
    head_offset = head_start * n_cols * n_cols
    col_offset = col_start * n_cols

    grad_output_base_ptr = grad_output_ptr + batch_offset + head_offset + col_offset
    output_base_ptr = output_ptr + batch_offset + head_offset + col_offset
    grad_input_base_ptr = grad_input_ptr + batch_offset + head_offset + col_offset

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    grad_output_ptrs = grad_output_base_ptr + col_offsets
    output_ptrs = output_base_ptr + col_offsets
    grad_input_ptrs = grad_input_base_ptr + col_offsets

    grad_output_row = tl.load(grad_output_ptrs, mask=mask, other=0.0)
    output_row = tl.load(output_ptrs, mask=mask, other=0.0)

    grad_input_row = output_row * (grad_output_row - tl.sum(output_row * grad_output_row, axis=0))
    tl.store(grad_input_ptrs, grad_input_row, mask=mask)

# Wrapper function for the forward pass
def softmax(X, mask=None, log=False, causal=False):
    n_batch, n_heads, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    num_stages = 2

    Y = torch.empty_like(X)

    if mask is None:
        mask_ptr = 0
    else:
        mask_ptr = mask

    _softmax[(n_batch * n_heads * n_cols, 1, 1)](
        Y, X, mask_ptr, n_batch, n_heads, n_cols, BLOCK_SIZE, log, causal, 1 if mask is not None else 0,
        num_warps=num_warps, num_stages=num_stages
    )
    return Y

# Wrapper function for the backward pass
def softmax_backward(grad_output, output):
    n_batch, n_heads, n_cols = grad_output.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    num_stages = 2

    grad_input = torch.empty_like(grad_output)

    _softmax_backward[(n_batch * n_heads * n_cols, 1, 1)](
        grad_output, output, grad_input, n_batch, n_heads, n_cols, BLOCK_SIZE,
        num_warps=num_warps, num_stages=num_stages
    )
    return grad_input

# Example usage
X = torch.randn(2, 3, 4).cuda()
mask = torch.randn(2, 3, 4).cuda()  # Optional mask
log = True  # Optional log transformation
causal = True  # Optional causal constraint

# Forward pass
Y = softmax(X, mask=mask, log=log, causal=causal)

# Backward pass
grad_output = torch.randn_like(Y).cuda()
grad_input = softmax_backward(grad_output, Y)
