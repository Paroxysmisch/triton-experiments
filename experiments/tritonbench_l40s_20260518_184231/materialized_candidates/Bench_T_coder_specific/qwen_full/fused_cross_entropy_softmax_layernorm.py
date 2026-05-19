import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, logits_row_stride,
    targets_ptr, targets_row_stride,
    output_prob_ptr, output_prob_row_stride,
    output_loss_ptr,
    N, C, K,
    normalized_shape,
    weight_ptr,
    ignore_index,
    reduction: tl.constexpr,
    label_smoothing: tl.constexpr,
    eps: tl.constexpr,
    elementwise_affine: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    targets_ptr += row_idx * targets_row_stride
    output_prob_ptr += row_idx * output_prob_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=float('-inf'))
    targets = tl.load(targets_ptr + col_offsets, mask=mask, other=ignore_index)

    if label_smoothing > 0.0:
        logits = logits * (1.0 - label_smoothing) + label_smoothing / C

    max_logits = tl.max(logits, 0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, 0) + eps

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        exp_logits = exp_logits * weight
        sum_exp_logits = sum_exp_logits * tl.sum(weight)

    probabilities = exp_logits / sum_exp_logits

    loss = tl.where(
        targets == ignore_index,
        0.0,
        -tl.log(tl.sum(tl.where(mask, probabilities, 0.0))),
    )

    if reduction == "mean":
        loss = loss / N
    elif reduction == "sum":
        pass
    # "none"

    tl.store(output_loss_ptr + row_idx, loss)

    if output_prob_ptr is not None:
        tl.store(output_prob_ptr + col_offsets, probabilities, mask=mask)

def fused_cross_entropy_softmax_layernorm(
    logits: Tensor,
    targets: Tensor,
    normalized_shape: int,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if elementwise_affine:
        gamma = torch.ones(normalized_shape, device=logits.device, dtype=logits.dtype)
        beta = torch.zeros(normalized_shape, device=logits.device, dtype=logits.dtype)
        layer_norm = lambda x: torch.nn.functional.layer_norm(
            x, normalized_shape, weight=gamma, bias=beta, eps=eps
        )
    else:
        layer_norm = lambda x: torch.nn.functional.layer_norm(
            x, normalized_shape, eps=eps
        )

    N, C = logits.shape[:2]
    K = targets.numel() // N

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if targets.stride(-1) != 1:
        targets = targets.contiguous()

    if out is None:
        out = torch.empty(
            (*targets.shape[: targets.ndim - K], C),
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        assert out.shape == targets.shape
        assert out.stride(-1) == 1

    loss = torch.empty(
        (*targets.shape[: targets.ndim - K], 1) if K > 1 else targets.shape[: K],
        dtype=logits.dtype,
        device=logits.device,
    )

    M = targets.numel() // K

    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (M, 1, 1)

    fused_cross_entropy_softmax_layernorm_kernel[grid](
        logits,
        logits.stride(0),
        targets,
        targets.stride(0),
        out,
        out.stride(0),
        loss,
        N,
        C,
        K,
        normalized_shape,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
        eps,
        elementwise_affine,
        M,
        BLOCK_SIZE,
    )

    if K == 1:
        loss = loss.squeeze()

    return loss, out
