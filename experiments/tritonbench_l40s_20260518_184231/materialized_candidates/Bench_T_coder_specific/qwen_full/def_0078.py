import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax_kernel(
    input_ptr, target_ptr, weight_ptr, output_ptr, n, c, ignore_index, label_smoothing,
    reduction: tl.constexpr, dim: tl.constexpr, ignore_index_set: tl.constexpr, 
    label_smoothing_set: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load
    target = tl.load(target_ptr + offsets, mask=mask, other=0)
    target_mask = mask & (target != ignore_index)

    input = tl.load(input_ptr + offsets * c, mask=mask)

    if dim != -1:
        dim = dim % c
        input = tl.trans(input, perm=[dim, (dim + 1) % 2])
    input = input.to(tl.float32)

    if label_smoothing_set:
        label_smoothing = label_smoothing.to(tl.float32)
        smooth = tl.full([BLOCK_SIZE], 1 - label_smoothing, dtype=tl.float32)
        smooth = smooth * label_smoothing / (c - 1)
    else:
        smooth = None

    # Fused log softmax and cross entropy
    loss, input = tl.log_softmax_and_cross_entropy(
        input, target, weight_ptr, ignore_index, label_smoothing, smooth
    )

    if ignore_index_set:
        loss = tl.where(target_mask, loss, 0)

    if reduction == "sum":
        loss = tl.sum(loss)
    elif reduction == "mean":
        if ignore_index_set:
            loss = tl.sum(loss) / tl.sum(target_mask.to(tl.float32))
        else:
            loss = tl.sum(loss) / n

    # Store
    tl.store(output_ptr + offsets, loss, mask=mask)


def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    assert input.ndim >= 2, "Input tensor must have at least 2 dimensions"
    assert (
        target.ndim == input.ndim - 1
    ), "Target tensor must have the same number of dimensions as input, except for the last dimension"
    assert 0 <= dim < input.ndim, "Invalid dimension"
    assert 0.0 <= label_smoothing < 1.0, "Label smoothing must be between 0 and 1"

    n = target.numel()
    c = input.shape[dim]
    assert (
        c == target.max() + 1
    ), "The class index of target must be in range [0, num_classes-1]"

    ignore_index_set = ignore_index != -100
    label_smoothing_set = label_smoothing > 0.0

    if weight is not None:
        assert weight.ndim == 1, "Weight must be a 1-dimensional tensor"
        assert weight.numel() == c, "Weight must have the same number of elements as the number of classes"
    else:
        weight = torch.ones((c,), dtype=torch.float32, device=input.device)

    if n <= 1024:
        BLOCK_SIZE = n
    else:
        BLOCK_SIZE = 1024

    loss = torch.empty((BLOCK_SIZE,), dtype=torch.float32, device=input.device)

    # Wrapper for triton kernel
    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    fused_cross_entropy_log_softmax_kernel[grid](
        input, target, weight, loss, n, c, ignore_index, label_smoothing, reduction, dim, ignore_index_set, label_smoothing_set, BLOCK_SIZE
    )

    return loss.sum()
