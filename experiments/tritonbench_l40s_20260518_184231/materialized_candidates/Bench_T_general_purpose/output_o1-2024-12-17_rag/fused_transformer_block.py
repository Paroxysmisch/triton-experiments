import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _layernorm_fwd_kernel(
    x_ptr,  # [M, N] input
    y_ptr,  # [M, N] output
    stride_x,  # stride for x
    stride_y,  # stride for y
    N,       # number of columns (normalized dimension)
    eps,     # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """
    A Triton kernel that applies layer normalization over the last dimension.
    Computation is done for one row per program_id(0).
    """
    row_id = tl.program_id(0)
    x_row_ptr = x_ptr + row_id * stride_x
    y_row_ptr = y_ptr + row_id * stride_y

    # 1) Compute mean and sum of squares in a first pass
    # We'll accumulate partial sums of x and x^2 in registers
    sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_x += x
        sum_x2 += x * x

    # Reduce partial sums across the BLOCK_SIZE vector
    mean_val = tl.sum(sum_x) / N
    mean_x2 = tl.sum(sum_x2) / N
    var_val = mean_x2 - mean_val * mean_val
    var_val = tl.where(var_val > 0.0, var_val, 0.0)
    rstd_val = 1.0 / tl.sqrt(var_val + eps)

    # 2) Apply normalization in a second pass
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean_val) * rstd_val
        tl.store(y_row_ptr + cols, x_hat, mask=mask)


def _layernorm_triton(input_tensor: torch.Tensor, eps: float):
    """
    Applies layer normalization over the last dimension of input_tensor using the Triton kernel.
    Gamma=1 and Beta=0 are assumed (i.e., no scale or bias).
    """
    # Flatten all dimensions except the last
    # Suppose input_tensor has shape (*, N), we flatten into [M, N]
    x_reshaped = input_tensor.reshape(-1, input_tensor.shape[-1])
    M, N = x_reshaped.shape
    # Prepare output
    y_reshaped = torch.empty_like(x_reshaped)

    # Determine BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(N)
    # Ensure BLOCK_SIZE is not excessively large:
    BLOCK_SIZE = max(32, min(BLOCK_SIZE, 1024))

    # Enqueue the Triton kernel
    grid = (M,)
    _layernorm_fwd_kernel[grid](
        x_reshaped,            # x_ptr
        y_reshaped,            # y_ptr
        x_reshaped.stride(0),  # stride_x
        y_reshaped.stride(0),  # stride_y
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to original
    return y_reshaped.reshape_as(input_tensor)


def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    """
    fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None) -> Tensor

    Performs a sequence of operations commonly used in a transformer block:

        1) Z1 = input @ weight1
        2) Z2 = softmax(Z1, dim=-1)
        3) Z3 = dropout(Z2, p=dropout_p)
        4) Z4 = Z3 @ weight2
        5) Y = LayerNorm(Z4 + residual, eps=eps)

    Args:
        input (Tensor):   Shape (*, N, D_in)
        weight1 (Tensor): Shape (D_in, D_k)
        weight2 (Tensor): Shape (D_k, D_out)
        residual (Tensor): Tensor broadcastable to shape of Z4
        dropout_p (float, optional): Probability of zeroing elements in dropout. Default: 0.1
        eps (float, optional): A value added to the denominator for numerical stability in layer normalization. Default: 1e-5
        out (Tensor, optional): Output tensor. Ignored if None. Default: None

    Returns:
        Tensor of shape (*, N, D_out) after the fused transformer operations.
    """
    # 1) MatMul
    z1 = input @ weight1

    # 2) Softmax along the last dimension
    z2 = F.softmax(z1, dim=-1)

    # 3) Dropout (enabled by default in this example)
    #    If dropout_p>0 and you want to disable during inference, set p=0 or adjust training=False here
    z3 = F.dropout(z2, p=dropout_p, training=True) if dropout_p > 0 else z2

    # 4) MatMul
    z4 = z3 @ weight2

    # 5) Add residual, then apply layer normalization
    z4_res = z4 + residual
    y = _layernorm_triton(z4_res, eps)

    if out is not None:
        out.copy_(y)
        return out
    return y
