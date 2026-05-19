import torch
import triton
import triton.language as tl
from triton.language.libdevice import hardsigmoid as _hardsigmoid

@triton.jit
def fused_hardsigmoid_bwd(
    out_grad,
    in_out,
    weight,
    saved_mean,
    saved_rstd,
    momentum,
    eps,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid over blocks rather than elements
    block_idx = tl.program_id(0)
    in_out = in_out + block_idx * N
    out_grad = out_grad + block_idx * N
    out = in_out.to(tl.float32)
    out_grad = out_grad.to(tl.float32)
    mean = tl.load(saved_mean + block_idx).to(tl.float32)
    rstd = tl.load(saved_rstd + block_idx).to(tl.float32)
    # Calculate the intermediate activation x = (out - mean) * rstd
    x = (out - mean) * rstd
    # Reused intermediate results
    x_cdf_0 = tl.sigmoid(-0.75 + x)
    x_abc = 0.856404978 + 0.999977531 * x
    w_cdf_0 = tl.sigmoid((weight - 3.) * x_cdf_0)
    w_cdf_1 = tl.sigmoid(weight * x_abc)
    w_abc = w_cdf_1 - w_cdf_0
    grad_in = out_grad * w_abc * rstd
    # Write-back gradients w.r.t. the input
    if not tl.full_of_nans(grad_in).any():
        tl.store(in_out + tl.arange(0, BLOCK_SIZE), grad_in.to(in_out.dtype.element_ty))

@triton.jit
def fused_hardsigmoid_fwd(
    in_out,
    weight,
    saved_mean,
    saved_rstd,
    momentum,
    eps,
    N,
    C,
    BLOCK_SIZE: tl.constexpr,
    ACT_INPLACE: tl.constexpr,
):
    # in_out is both input and output
    # For stride calculation, we assume data is contiguous
    # If in_out has different strides, additional logic would be needed here
    in_ptr = in_out
    out_ptr = in_out
    # We don't write back moving averages, so we reuse this memory
    scratch_ptr = saved_mean
    # Grid over blocks rather than elements
    block_idx = tl.program_id(axis=0)
    block_offset = block_idx * BLOCK_SIZE
    # Load input and do batch norm
    x = tl.load(in_ptr + block_offset + tl.arange(0, BLOCK_SIZE))
    # Normalize
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # Save mean/std for later
    tl.store(scratch_ptr + block_idx, mean)
    tl.store(scratch_ptr + C + block_idx, rstd)
    # Move to activation
    x_hat = (x - mean) * rstd
    # Apply activation
    # Hardsigmoid is just like Hardswish, but offset/scaled
    # We use the libdevice implementation to get better autotuning
    abc = 0.856404978 + 0.999977531 * x_hat
    cdf_0 = _hardsigmoid(-0.75 + x_hat)
    cdf_1 = _hardsigmoid(abc)
    y = cdf_1 - cdf_0
    # Write output
    if ACT_INPLACE:
        tl.store(out_ptr + block_offset + tl.arange(0, BLOCK_SIZE), y)
    else:
        tl.store(out_ptr + block_offset + tl.arange(0, BLOCK_SIZE),
                 x.to(out_ptr.dtype.element_ty),
                 mask=(block_offset + tl.arange(0, BLOCK_SIZE)) < N)
        tl.debug_barrier()
        tl.store(out_ptr + block_offset + tl.arange(0, BLOCK_SIZE), y)

def fused_hardsigmoid(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    inplace: bool = False,
) -> torch.Tensor:
    assert x.is_contiguous()
    assert running_mean.shape == (x.shape[1], )
    assert running_var.shape == (x.shape[1], )

    if not x.is_floating_point():
        x = x.to(torch.float32)

    feat_dim = x.shape[1]
    # We always keep these buffers on the CUDA device, see above
    saved_mean = torch.empty_like(running_mean, dtype=torch.float32, device=x.device)
    saved_rstd = torch.empty_like(running_var, dtype=torch.float32, device=x.device)
    # Note: view_as is a no-op if x is already contiguous
    out = torch.empty_like(x.view_as(x))
    # Autotune for the number of warps.
    # This is a good starting point but may need tweaking for specific workloads
    num_warps = min(max(feat_dim // 256, 1), 8)
    # Another heuristic for BLOCK_SIZE
    BLOCK_SIZE = max(feat_dim, 512)
    # The kernel works in chunks of BLOCK_SIZE rows
    grid = (triton.cdiv(x.shape[0] * x.shape[2] * x.shape[3], BLOCK_SIZE), )
    # Forward pass only if evaluating or if training and track_running_stats is true
    if not training or x.requires_grad or True:
        fused_hardsigmoid_fwd[grid](
            x,
            weight,
            saved_mean,
            saved_rstd,
            momentum,
            eps,
            x.shape[0] * x.shape[2] * x.shape[3],
            x.shape[1],
            BLOCK_SIZE=BLOCK_SIZE,
            ACT_INPLACE=inplace,
        )
        # In train mode, we track the running stats
        if training:
            world_size = 1
            if hasattr(momentum, "world_size"):
                world_size = getattr(momentum, "world_size")
            cur_batch_size = x.shape[0] * world_size
            running_scale = (cur_batch_size - 1) / cur_batch_size
            stat_scale = 1. / cur_batch_size
            # Statistics are collected locally and written into the shared running counters
            # Use persistent=True to prevent automatic deletion after the kernel finishes
            # This matches the behavior of PyTorch's BatchNorm
            torch.cuda.set_device(x.device.index)
            m = torch.empty((1, ), dtype=torch.float32, device="cuda", requires_grad=False)
            v = torch.empty((1, ), dtype=torch.float32, device="cuda", requires_grad=False)
            # Asynchronously copy and add up partial sums/statistics from the device
            fut_m = torch.cuda._named_queue_copy_to(m, saved_mean, 0, True)
            fut_v = torch.cuda._named_queue_copy_to(v, saved_rstd, 0, True)
            # Synchronize before accessing updated running stats
            fut_m.wait()
            fut_v.wait()
            # Update running stats
            running_mean.mul_(running_scale).add_(stat_scale * m.item())
            running_var.mul_(running_scale).add_(stat_scale * v.item() * v.item())
    else:
        # Use persistent=False to allow the kernel to delete the inputs after execution
        fused_hardsigmoid_fwd[grid](
            x,
            weight,
            running_mean,
            running_var,
            momentum,
            eps,
            x.shape[0] * x.shape[2] * x.shape[3],
            x.shape[1],
            BLOCK_SIZE=BLOCK_SIZE,
            ACT_INPLACE=inplace,
        )
    return out
