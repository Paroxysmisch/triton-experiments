import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def fused_cross_entropy_softmax_layernorm(
    logits: torch.Tensor,
    targets: torch.Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute cross-entropy loss
    if targets.shape == logits.shape:
        # Targets are class probabilities
        log_probs = F.log_softmax(logits, dim=1)
        loss = - (targets * log_probs).sum(dim=1)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        # Targets are class indices
        loss = F.cross_entropy(
            logits, targets, weight=weight, ignore_index=ignore_index,
            reduction=reduction, label_smoothing=label_smoothing
        )
    
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # Apply layer normalization without affine transformation
    normalized_probs = F.layer_norm(
        probs, normalized_shape=normalized_shape, weight=None, bias=None, eps=eps
    )
    
    # Output handling
    if out is not None:
        out.copy_(normalized_probs)
    
    return loss, normalized_probs

@triton.jit
def _fused_kernel(
    logits_ptr, targets_ptr, output_ptr, loss_ptr,
    n_classes, n_rows, eps,
    logits_stride, targets_stride, output_stride, loss_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= n_rows:
        return
    
    # Offsets for the current row
    row_offset_logits = pid * logits_stride
    row_offset_output = pid * output_stride
    target_idx = tl.load(targets_ptr + pid * targets_stride)
    
    # Load logits for the current row
    cols = tl.arange(0, BLOCK_SIZE)
    logits_ptrs = logits_ptr + row_offset_logits + cols
    mask = cols < n_classes
    logits = tl.load(logits_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    max_logit = tl.max(logits, axis=0)
    logits -= max_logit
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp
    
    # Compute cross-entropy loss
    if target_idx >= 0 and target_idx < n_classes:
        target_prob = tl.load(logits_ptr + row_offset_logits + target_idx)
        loss = -tl.log(probs[target_idx])
    else:
        loss = 0.0
    tl.store(loss_ptr + pid * loss_stride, loss, mask=mask)
    
    # Compute layer normalization
    mean = tl.sum(probs) / n_classes
    var = tl.sum((probs - mean) ** 2) / n_classes
    normalized = (probs - mean) / tl.sqrt(var + eps)
    
    # Store normalized probabilities
    output_ptrs = output_ptr + row_offset_output + cols
    tl.store(output_ptrs, normalized, mask=mask)

def triton_fused_cross_entropy_softmax_layernorm(
    logits: torch.Tensor,
    targets: torch.Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    *,
    out: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Simplified Triton kernel for 2D logits and 1D targets (class indices)
    assert logits.dim() == 2, "Triton kernel only supports 2D logits"
    assert targets.dim() == 1, "Triton kernel only supports 1D targets"
    N, C = logits.shape
    device = logits.device
    
    # Allocate outputs
    loss = torch.empty(N, device=device)
    normalized_probs = torch.empty_like(logits)
    if out is not None:
        assert out.shape == logits.shape, "Output tensor shape mismatch"
        normalized_probs = out
    
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    # Launch kernel
    grid = (N,)
    _fused_kernel[grid](
        logits, targets, normalized_probs, loss,
        C, N, eps,
        logits.stride(0), targets.stride(0), normalized_probs.stride(0), loss.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Apply reduction
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss, normalized_probs
