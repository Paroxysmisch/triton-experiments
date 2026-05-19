import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# This kernel is based on PyTorch's _sgd_step_fused.
@triton.jit
def _fused_sgd_step_kernel(
    p_ptr,
    grad_ptr,
    momentum_buffer_ptr,
    velocity_ptr,
    lr,
    wd,
    momentum,
    dampening,
    nesterov,
    use_nesterov_momentum,
    max_lr,
    min_lr,
    lr_scale,
    lr_bias_correction_exponent,
    bias_correction_exponent_offset,
    n_elements,
    BLOCK_SIZE,  # tl.constexpr
):
    """
    Kernel for computing a single step of Stochastic Gradient Descent (SGD).
    Unlike the standard form of SGD, this kernel uses only a single momentum
    buffer and velocity accumulator per parameter, which reduces memory usage
    and allows for greater flexibility in terms of the order of operations
    within a step.
    """
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_momentum_buffer_ptr = momentum_buffer_ptr + offsets
    offset_velocity_ptr = velocity_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    momentum_buffer = tl.load(offset_momentum_buffer_ptr, mask=mask)
    velocity = tl.load(offset_velocity_ptr, mask=mask)

    # Step weight decay
    if wd != 0:
        p = p * (1 - lr * wd)

    # Compute the effective momentum coefficient and apply it to the momentum buffer.
    # Note that we allow momentum to be zero for the first step; this effectively
    # makes the momentum buffer act like a velocity accumulator.
    mom_coeff = (
        momentum
        * (1 - (lr * dampening))
        / ((1 - lr) ** lr_bias_correction_exponent + bias_correction_exponent_offset)
    )
    if nesterov:
        momentum_buffer = momentum_buffer * mom_coeff + grad
    else:
        momentum_buffer = momentum_buffer * mom_coeff + grad * (1 - lr)

    # Compute the update scale and apply it to the momentum buffer.
    update_scale = lr * ((1 - momentum) / (1 - mom_coeff))
    if max_lr != 0:
        update_scale = min(max_lr, update_scale)
    if min_lr != 0:
        update_scale = max(min_lr, update_scale)
    update_scale *= lr_scale
    momentum_buffer *= update_scale

    # Store new momentum buffer and compute new param value
    if nesterov and use_nesterov_momentum:
        p = p + momentum_buffer * mom_coeff + grad
    else:
        p = p + momentum_buffer
    tl.store(offset_p_ptr, p, mask=mask)
    # Store new velocity
    velocity = momentum_buffer / (1 - mom_coeff)
    tl.store(offset_velocity_ptr, velocity, mask=mask)
    # Store new momentum buffer
    tl.store(offset_momentum_buffer_ptr, momentum_buffer, mask=mask)

def sgd_fused_triton(
    params: torch.Tensor,
    grads: torch.Tensor,
    momentum_buffer: torch.Tensor,
    velocities: torch.Tensor,
    lr: float,
    wd: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    use_nesterov_momentum: bool,
    max_lr: float,
    min_lr: float,
    lr_scale: float,
    lr_bias_correction_exponent: float,
    bias_correction_exponent_offset: float,
) -> None:
    assert all([t.is_cuda for t in (params, grads, momentum_buffer, velocities)])
    n_elements = volume(params.shape)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _fused_sgd_step_kernel[grid](
        params,
        grads,
        momentum_buffer,
        velocities,
        lr,
        wd,
        momentum,
        dampening,
        nesterov,
        use_nesterov_momentum,
        max_lr,
        min_lr,
        lr_scale,
        lr_bias_correction_exponent,
        bias_correction_exponent_offset,
        n_elements,
    )
