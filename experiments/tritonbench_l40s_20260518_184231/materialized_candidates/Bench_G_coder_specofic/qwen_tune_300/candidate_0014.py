import torch
import triton
import triton.language as tl
from flag_gems.utils import dist_util

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride,
    labels_ptr,
    losses_ptr, lse_ptr,
    smoothing, lse_square_scale,
    ignored_index,
    logits_dim, n_valid_logits,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    logits_start_ptr = logits_ptr + pid * logits_row_stride
    labels_start_ptr = labels_ptr + pid
    losses_start_ptr = losses_ptr + pid
    lse_start_ptr = lse_ptr + pid

    logits_offsets = tl.arange(0, BLOCK_SIZE)
    labels_offset = 0
    losses_offset = tl.arange(0, BLOCK_SIZE)
    lse_offset = tl.arange(0, BLOCK_SIZE)

    labels_val = tl.load(labels_start_ptr + labels_offset, mask=labels_offset < n_valid_logits, other=ignored_index)
    is_ignored = labels_val == ignored_index

    logits_vals = tl.load(logits_start_ptr + logits_offsets, mask=logits_offsets < n_valid_logits, other=-float("inf"))
    logits_val = tl.where(labels_val[:, None] == tl.arange(0, BLOCK_SIZE)[None, :], logits_vals, -float("inf"))

    lse_val = tl.max(logits_val, axis=0)
    numerators = tl.exp(logits_val - lse_val)
    denominators = tl.sum(numerators)

    if smoothing > 0:
        smooth_loss = -tl.log(tl.sum(tl.exp(logits_vals - lse_val)) / n_valid_logits)
        smooth_loss *= smoothing * (1 - smoothing) / 2
        loss_val = tl.where(is_ignored, 0.0, (lse_val + tl.log(denominators) + smooth_loss))
    else:
        loss_val = tl.where(is_ignored, 0.0, lse_val + tl.log(denominators))

    if lse_square_scale > 0:
        loss_val += lse_square_scale * lse_val * lse_val

    tl.store(losses_start_ptr + losses_offset, loss_val, mask=losses_offset < n_valid_logits)
    tl.store(lse_start_ptr + lse_offset, lse_val, mask=lse_offset < n_valid_logits)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride,
    labels_ptr,
    losses_ptr, lse_ptr,
    gradients_ptr, logits_dim,
    smoothing, lse_square_scale,
    ignored_index,
    logits_dim, n_valid_logits,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    logits_start_ptr = logits_ptr + pid * logits_row_stride
    labels_start_ptr = labels_ptr + pid
    losses_start_ptr = losses_ptr + pid
    lse_start_ptr = lse_ptr + pid
    gradients_start_ptr = gradients_ptr + pid * logits_row_stride

    logits_offsets = tl.arange(0, BLOCK_SIZE)
    labels_offset = 0
    gradients_offsets = tl.arange(0, BLOCK_SIZE)

    labels_val = tl.load(labels_start_ptr + labels_offset, mask=labels_offset < n_valid_logits, other=ignored_index)
    is_ignored = labels_val == ignored_index

    logits_vals = tl.load(logits_start_ptr + logits_offsets, mask=logits_offsets < n_valid_logits, other=float("inf"))
    logits_val = tl.where(labels_val[:, None] == tl.arange(0, BLOCK_SIZE)[None, :], logits_vals, float("inf"))

    lse_val = tl.load(lse_start_ptr + labels_offset, mask=labels_offset < n_valid_logits)
    loss_val = tl.load(losses_start_ptr + labels_offset, mask=labels_offset < n_valid_logits)
    loss_gradient_val = tl.exp(logits_val - lse_val) / tl.exp(lse_val) - 1

    if smoothing > 0:
        smooth_loss_gradient = smoothing * (1 - smoothing) / 2
        smooth_loss_gradient *= tl.where(
            is_ignored[:, None], 0.0, tl.exp(logits_val - lse_val) / tl.exp(lse_val)
        )
        loss_gradient_val += smooth_loss_gradient

    if lse_square_scale > 0:
        loss_gradient_val += 2 * lse_square_scale * lse_val * tl.exp(logits_val - lse_val) / tl.exp(lse_val)

    loss_gradient_val = tl.where(is_ignored[:, None], 0.0, loss_gradient_val)
    logits_gradient_vals = loss_gradient_val * tl.exp(logits_val - lse_val)

    tl.store(gradients_start_ptr + gradients_offsets, logits_gradient_vals, mask=gradients_offsets < n_valid_logits)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits, labels,
        smoothing=0, lse_square_scale=0,
        ignored_index=-100,
        process_group=None,
    ):
        n_valid_logits = logits.shape[-1]
        assert labels.shape[0] == logits.shape[0]
        assert labels.dtype in (torch.int16, torch.int32, torch.int64)

        if process_group is not None:
            world_size = dist_util.get_world_size(process_group)
            rank = dist_util.get_rank(process_group)
        else:
            world_size = 1
            rank = 0

        if world_size > 1:
            assert ignored_index != -100

        device = logits.device
        dtype = logits.dtype
        BLOCK_SIZE = triton.next_power_of_2(n_valid_logits)

        comm_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
        losses = torch.empty(labels.shape, dtype=comm_dtype, device=device)
        lse = torch.empty(labels.shape, dtype=comm_dtype, device=device)

        logits_dim = logits.shape[-1]
        logits = logits.reshape(-1, logits_dim)
        labels = labels.reshape(-1)
        n_valid_logits = (labels != ignored_index).sum().item()
        num_blocks = labels.shape[0]

        with torch.cuda.device(device):
            cross_entropy_fwd_kernel[(num_blocks,)](
                logits, logits.strides[-2],
                labels,
                losses, lse,
                smoothing, lse_square_scale,
                ignored_index,
                logits_dim, n_valid_logits,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        if world_size > 1:
            process_group = process_group if process_group is not None else torch.distributed.get_world_group()
            token_losses = torch.empty(labels.shape, dtype=comm_dtype, device=device)
            token_lse = torch.empty(labels.shape, dtype=comm_dtype, device=device)

            dist.all_reduce(losses, op=dist.ReduceOp.SUM, group=process_group)
            dist.all_reduce(lse, op=dist.ReduceOp.MAX, group=process_group)

            token_losses[labels != ignored_index] = losses[labels != ignored_index] / world_size
            token_lse[labels != ignored_index] = lse[labels != ignored_index]

            losses = token_losses
            lse = token_lse

        ctx.save_for_backward(logits, labels, losses, lse)
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.process_group = process_group

        return losses

    @staticmethod
    def backward(ctx, losses_grad):
        logits, labels, losses, lse = ctx.saved_tensors

        n_valid_logits = (labels != ctx.ignored_index).sum().item()

        if ctx.process_group is not None:
            world_size = dist_util.get_world_size(ctx.process_group)
        else:
            world_size = 1

        device = logits.device
        BLOCK_SIZE = triton.next_power_of_2(logits.shape[-1])

        comm_dtype = logits.dtype if logits.dtype in (torch.float16, torch.bfloat16) else torch.float32
        logits_grad = torch.empty_like(logits, dtype=comm_dtype)
        gradients = torch.empty(labels.shape + (logits.shape[-1],), dtype=comm_dtype, device=device)

        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        num_blocks = labels.shape[0]

        with torch.cuda.device(device):
            cross_entropy_bwd_kernel[(num_blocks,)](
                logits, logits.strides[-2],
                labels,
                losses, lse,
                gradients, logits.shape[-1],
                ctx.smoothing, ctx.lse_square_scale,
                ctx.ignored_index,
                logits.shape[-1], n_valid_logits,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        logits_grad = gradients.reshape_as(logits)

        if world_size > 1:
            process_group = ctx.process_group if ctx.process_group is not None else torch.distributed.get_world_group()

            dist.all_reduce(logits_grad, op=dist.ReduceOp.SUM, group=process_group)

        return logits_grad, None, None, None, None, None

def cross_entropy_loss(
    logits, labels,
    smoothing=0, lse_square_scale=0,
    ignored_index=-100,
    process_group=None,
):
    return CrossEntropyLoss.apply(
        logits, labels,
        smoothing, lse_square_scale,
        ignored_index,
        process_group,
    )
