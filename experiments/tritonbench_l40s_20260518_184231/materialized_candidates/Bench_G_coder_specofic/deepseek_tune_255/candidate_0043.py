import torch
import triton
import triton.language as tl
from triton import next_power_of_2

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["V"],
)
@triton.heuristics({'BLOCK_SIZE': lambda args: args['V']})
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, log_target, loss_ptr,
    stride_y_pred_batch, stride_y_pred_v,
    stride_y_true_batch, stride_y_true_v,
    n_batch, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)

    # Create offsets for predicted and true values
    y_pred_ptr += batch_idx * stride_y_pred_batch
    y_true_ptr += batch_idx * stride_y_true_batch

    # Create offsets for columns
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load predicted and true values
    y_pred = tl.load(y_pred_ptr + col_offsets * stride_y_pred_v, mask=col_offsets < V, other=-float('inf'))
    y_true = tl.load(y_true_ptr + col_offsets * stride_y_true_v, mask=col_offsets < V, other=-float('inf'))

    # Calculate KL divergence
    if log_target:
        loss = y_true * (tl.log(y_true + eps) - tl.log(tl.exp(y_pred) + eps))
    else:
        loss = y_true * (tl.log(y_true + eps) - y_pred)

    # Sum losses
    loss = tl.sum(loss)

    # Store or accumulate loss
    if batch_idx == 0:
        tl.store(loss_ptr, loss)
    else:
        loss_accum = tl.load(loss_ptr)
        loss_accum += loss
        tl.store(loss_ptr, loss_accum)


def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="mean", eps=1e-7):
    n_batch, V = y_true.shape

    # Create output tensor
    if reduction == "none":
        loss = torch.empty((n_batch, V), device=y_true.device, dtype=torch.float32)
    else:
        loss = torch.tensor(0.0, device=y_true.device, dtype=torch.float32)

    # Move tensors to CUDA
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()

    # Ensure inputs are CUDA
    assert y_pred.is_cuda and y_true.is_cuda

    # Kernel configuration
    grid = (n_batch,)
    _kldiv_kernel_forward[grid](
        y_pred, y_true, log_target, loss,
        y_pred.stride(0), y_pred.stride(1),
        y_true.stride(0), y_true.stride(1),
        n_batch, V,
        eps=eps,
    )

    # Apply reduction
    if reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "batchmean":
        loss = torch.sum(loss) / n_batch

    return loss


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["V"],
)
@triton.heuristics({'BLOCK_SIZE': lambda args: args['V']})
@triton.jit
def _kldiv_kernel_backward(
    target_ptr, grad_output_ptr, new_grads_ptr, log_target,
    stride_target_batch, stride_target_v,
    n_batch, V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)

    # Create offsets for target and grad_output
    target_ptr += batch_idx * stride_target_batch
    grad_output_ptr += batch_idx * stride_target_batch

    # Create offsets for columns
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load target and grad_output
    target = tl.load(target_ptr + col_offsets * stride_target_v, mask=col_offsets < V, other=0)
    grad_output = tl.load(grad_output_ptr + col_offsets * stride_target_v, mask=col_offsets < V, other=0)

    # Calculate gradients
    if log_target:
        grad = tl.exp(target) * grad_output
    else:
        grad = target * (tl.log(target) - grad_output)

    # Sum gradients
    grad = tl.sum(grad)

    # Store gradients
    tl.store(new_grads_ptr + batch_idx, grad)


def kldiv_backward_triton(target, grad_output, new_grads, log_target=False):
    n_batch, V = target.shape

    # Ensure inputs are CUDA
    assert target.is_cuda and grad_output.is_cuda

    # Create output tensor
    new_grads = torch.empty((n_batch,), device=target.device, dtype=torch.float32)

    # Kernel configuration
    grid = (n_batch,)
    _kldiv_kernel_backward[grid](
        target, grad_output, new_grads, log_target,
        target.stride(0), target.stride(1),
        n_batch, V,
    )

    return new_grads
