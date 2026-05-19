import torch
from typing import Tuple

def fused_cross_entropy_softmax_layernorm(
    logits: torch.Tensor,
    targets: torch.Tensor,
    normalized_shape: int or list or torch.Size,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    out: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate inputs
    assert isinstance(logits, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert isinstance(normalized_shape, (int, list, torch.Size))
    assert weight is None or isinstance(weight, torch.Tensor)
    assert isinstance(ignore_index, int)
    assert reduction in ['none', 'mean', 'sum']
    assert isinstance(label_smoothing, float)
    assert isinstance(eps, float)
    assert out is None or isinstance(out, torch.Tensor)

    # Convert normalized_shape to tuple
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    elif isinstance(normalized_shape, list):
        normalized_shape = tuple(normalized_shape)

    # Get shapes and device
    batch_size, num_classes = logits.shape[:2]
    num_elements = logits.numel()
    normalized_shape_size = len(normalized_shape)
    weight_size = weight.numel() if weight is not None else 0

    # Create output tensors
    if out is None:
        out = torch.zeros_like(logits[:, :1])
    log_probs = torch.zeros_like(logits)

    # Allocate memory for Triton kernel
    logits_device = logits.device
    outputs = {
        'output': out,
        'log_probs': log_probs,
        'ln_mean': torch.zeros(batch_size, dtype=logits.dtype, device=logits_device),
        'ln_var': torch.zeros(batch_size, dtype=logits.dtype, device=logits_device),
        'ln_out': torch.zeros_like(logits),
    }

    # Launch Triton kernel
    grid = (num_elements // BLOCK_SIZE + 1, 1)
    fused_cross_entropy_softmax_layernorm_kernel[
        grid,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    ](
        logits.data_ptr(),
        targets.data_ptr(),
        outputs['log_probs'].data_ptr(),
        outputs['ln_mean'].data_ptr(),
        outputs['ln_var'].data_ptr(),
        outputs['ln_out'].data_ptr(),
        weight.data_ptr() if weight is not None else 0,
        normalized_shape_size,
        weight_size,
        ignore_index,
        reduction,
        label_smoothing,
        eps,
        num_elements,
        num_classes,
        normalized_shape_size,
        weight_size,
    )

    return outputs['output'], outputs['ln_out']
