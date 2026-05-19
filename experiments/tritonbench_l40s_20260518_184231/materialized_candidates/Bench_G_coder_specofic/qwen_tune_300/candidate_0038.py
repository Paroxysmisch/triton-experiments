import torch
import triton
import triton.language as tl
from .softmax_kernels import get_num_warps

def cfggen():
    block_size = [16, 32, 64, 128]
    warps = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_SIZE_H": bs}, num_warps=w)
        for bs in block_size
        for w in warps
    ]
    return configs

@triton.jit
def cross_entropy_fwd_kernel(
    logits,  # logits tensor
    labels,  # labels tensor
    loss,  # loss tensor
    z_loss,  # z_loss tensor
    lse,  # lse tensor
    smoothing: tl.constexpr,  # smoothing factor
    logit_scale: tl.constexpr,  # logit scale factor
    lse_square_scale: tl.constexpr,  # lse square scale factor
    ignored_index: tl.constexpr,  # index of ignored classes
    total_classes: tl.constexpr,  # total number of classes
    class_start_idx: tl.constexpr,  # starting index of classes for this program
    BLOCK_SIZE_H: tl.constexpr,  # block size height
    BLOCK_SIZE_W: tl.constexpr,  # block size width
):
    program_idx = tl.program_id(0)
    logits = logits + program_idx * total_classes
    labels = labels + program_idx
    loss = loss + program_idx
    z_loss = z_loss + program_idx
    lse = lse + program_idx
    class_offsets = tl.arange(0, BLOCK_SIZE_H) + class_start_idx
    class_mask = class_offsets < total_classes
    labels_offset = tl.load(labels) + class_start_idx
    logits_offset = tl.load(logits + labels_offset) * logit_scale
    logits_block_ptr = logits + class_offsets
    logits = tl.load(logits_block_ptr, mask=class_mask, other=-float("inf"))
    logits = logits * logit_scale
    lse_val = tl.max(logits, 0)
    if smoothing > 0:
        logits = logits - lse_val
        exp_logits = tl.exp(logits)
        probs = exp_logits / tl.sum(exp_logits, 0)
        xent = tl.log(tl.sum(probs * exp_logits, 0)) - lse_val
        probs = probs * (1 - smoothing) + smoothing / total_classes
        lse_square = tl.sum(-probs * tl.log(probs), 0)
    else:
        logits = logits - lse_val
        xent = tl.log(tl.exp(logits) / total_classes) - lse_val
    lse_val = lse_val.to(tl.float32)
    xent = xent.to(tl.float32)
    if labels_offset == class_start_idx:
        tl.store(loss, xent)
    else:
        tl.atomic_add(loss, xent)
    if smoothing > 0:
        if labels_offset == class_start_idx:
            tl.store(z_loss, lse_square)
        else:
            tl.atomic_add(z_loss, lse_square)
        lse_val = lse_val + lse_square_scale * lse_square
    if labels_offset != class_start_idx and lse_val != -float("inf"):
        tl.store(lse, lse_val)

@triton.jit
def cross_entropy_bwd_kernel(
    logits,  # logits tensor
    dlogits,  # dlogits tensor
    loss,  # loss tensor
    lse,  # lse tensor
    labels,  # labels tensor
    smoothing: tl.constexpr,  # smoothing factor
    logit_scale: tl.constexpr,  # logit scale factor
    ignored_index: tl.constexpr,  # index of ignored classes
    total_classes: tl.constexpr,  # total number of classes
    class_start_idx: tl.constexpr,  # starting index of classes for this program
    BLOCK_SIZE_H: tl.constexpr,  # block size height
    BLOCK_SIZE_W: tl.constexpr,  # block size width
):
    program_idx = tl.program_id(0)
    logits = logits + program_idx * total_classes
    dlogits = dlogits + program_idx * total_classes
    labels = labels + program_idx
    loss = loss + program_idx
    lse = lse + program_idx
    class_offsets = tl.arange(0, BLOCK_SIZE_H) + class_start_idx
    class_mask = class_offsets < total_classes
    labels_offset = tl.load(labels) + class_start_idx
    logits_offset = tl.load(logits + labels_offset) * logit_scale
    logits_block_ptr = logits + class_offsets
    logits = tl.load(logits_block_ptr, mask=class_mask, other=-float("inf"))
    logits = logits * logit_scale
    lse_val = tl.load(lse).to(tl.float32)
    probs = tl.exp(logits - lse_val)
    probs = probs / tl.sum(probs, 0)
    if smoothing > 0:
        probs = probs * (1 - smoothing) + smoothing / total_classes
    dloss = tl.load(loss)
    probs = probs.to(tl.float32)
    dlogits_offset = probs * dloss
    if labels_offset == class_start_idx:
        dlogits_offset = dlogits_offset - dloss
    tl.store(dlogits_block_ptr, dlogits_offset, mask=class_mask)

def cross_entropy_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    logit_scale: float,
    lse_square_scale: float,
    ignored_index: int,
    total_classes: int,
    class_start_idx: int = 0,
    BLOCK_SIZE: int = 16,
    HAS_SMOOTHING: bool = True,
    SPLIT: bool = False,
):
    num_rows = labels.shape[0]
    num_classes = logits.shape[1] // num_rows
    if not SPLIT:
        grid = (num_rows,)
        cross_entropy_fwd_kernel[grid](
            logits,
            labels,
            smoothing,
            logit_scale,
            lse_square_scale,
            ignored_index,
            total_classes,
            class_start_idx,
            num_classes,
            BLOCK_SIZE_H=triton.next_power_of_2(num_classes),
            BLOCK_SIZE_W=16,
            num_stages=1,
            num_warps=get_num_warps(num_classes),
        )
        loss = logits.new_zeros(num_rows)
        z_loss = logits.new_zeros(num_rows) if HAS_SMOOTHING else None
        lse = logits.new_zeros(num_rows)
        return loss, lse, z_loss
    else:
        num_warps = get_num_warps(BLOCK_SIZE)
        num_stages = 1
        grid = (num_rows, triton.cdiv(total_classes, BLOCK_SIZE))
        cross_entropy_fwd_kernel[grid](
            logits,
            labels,
            smoothing,
            logit_scale,
            lse_square_scale,
            ignored_index,
            total_classes,
            class_start_idx,
            BLOCK_SIZE_H=BLOCK_SIZE,
            BLOCK_SIZE_W=16,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        loss = torch.zeros((num_rows,), dtype=torch.float32, device=logits.device)
        z_loss = (
            torch.zeros((num_rows,), dtype=torch.float32, device=logits.device)
            if HAS_SMOOTHING
            else None
        )
        lse = torch.zeros((num_rows,), dtype=torch.float32, device=logits.device)
        return loss, lse, z_loss

def cross_entropy_bwd(
    logits: torch.Tensor,
    dlogits: torch.Tensor,
    loss: torch.Tensor,
    lse: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    logit_scale: float,
    ignored_index: int,
    total_classes: int,
    class_start_idx: int = 0,
    BLOCK_SIZE: int = 16,
    HAS_SMOOTHING: bool = True,
    SPLIT: bool = False,
):
    num_rows = labels.shape[0]
    num_classes = logits.shape[1] // num_rows
    if not SPLIT:
        grid = (num_rows,)
        cross_entropy_bwd_kernel[grid](
            logits,
            dlogits,
            loss,
            lse,
            labels,
            smoothing,
            logit_scale,
            ignored_index,
            total_classes,
            class_start_idx,
            num_classes,
            BLOCK_SIZE_H=triton.next_power_of_2(num_classes),
            BLOCK_SIZE_W=16,
            num_stages=1,
            num_warps=get_num_warps(num_classes),
        )
        return dlogits
    else:
        num_warps = get_num_warps(BLOCK_SIZE)
        num_stages = 1
        grid = (num_rows, triton.cdiv(total_classes, BLOCK_SIZE))
        cross_entropy_bwd_kernel[grid](
            logits,
            dlogits,
            loss,
            lse,
            labels,
            smoothing,
            logit_scale,
            ignored_index,
            total_classes,
            class_start_idx,
            BLOCK_SIZE_H=BLOCK_SIZE,
            BLOCK_SIZE_W=16,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return dlogits
