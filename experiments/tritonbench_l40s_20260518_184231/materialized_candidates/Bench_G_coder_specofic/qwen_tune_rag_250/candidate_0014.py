tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)


@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits,  # logits is always required
        labels,  # labels is always required
        smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignored_index=-100,
        process_group=None,
    ):
        if labels.dtype == torch.long and labels.data_ptr() % 16 != 0:
            labels = F.pad(labels, (0, 1))[..., :-1]
            assert labels.data_ptr() % 16 == 0
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols
        if hasattr(logits, "tensor_parallel_size"):
            assert (
                logits.tensor_parallel_size == world_size
            ), "Tensor parallel size must match the group size for the loss"
            logits.tensor_parallel_size = world_size
        else:
            logits.tensor_parallel_size = world_size
        if hasattr(labels, "tensor_parallel_size"):
            assert (
                labels.tensor_parallel_size == world_size
            ), "Tensor parallel size must match the group size for the loss"
            labels.tensor_parallel_size = world_size
        else:
            labels.tensor_parallel_size = world_size
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 8192:
            num_warps = 32
        elif BLOCK_SIZE >= 4096:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        lse_square_scale = float(lse_square_scale)
        if ctx.save_dlogits:
            dlogits = torch.empty_like(logits)
            ctx.save_for_backward(dlogits, logits, labels)
        else:
            ctx.save_for_backward(logits, labels)
        n_splits = triton.cdiv(n_rows, BLOCK_SIZE)
        loss = torch.empty(n_splits, n_rows, dtype=torch.float, device=logits.device)
        lse = torch.empty(n_splits, n_rows, dtype=torch.float, device=logits.device)
        z_loss = torch.empty(n_splits, n_rows, dtype=torch.float, device=logits.device)
        grid = (n_rows, n_splits)
        with torch.cuda.device(rank):
            cross_entropy_fwd_kernel[grid](
                loss,  # data ptrs
                lse,
                z_loss,
                logits,
                labels,
                smoothing,
                logit_scale,
                lse_square_scale,
                ignored_index,
                total_classes,
                class_start_idx,
                n_cols,  # shapes
                n_rows,
                logits.stride(0),  # strides
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
                SPLIT=n_splits > 1,
            )
        if n_splits > 1:
            if process_group is not None:
                split_loss = all_reduce(loss, process_group=process_group)
                split_z_loss = all_reduce(z_loss, process_group=process_group)
                split_lse = all_reduce(lse, process_group=process_group)
            else:
                split_loss = gather(loss, 1, process_group=process_group)
                split_z_loss = gather(z_loss, 1, process_group=process_group)
                split_lse = gather(lse, 1, process_group=process_group)
            final_lse = (
                (split_lse * torch.exp(split_z_loss) - 2.0 * split_z_loss * torch.sum(torch.exp(split_lse))
                ) / (n_rows * total_classes)
            )
            final_loss = split_loss + final_lse
        else:
            final_lse = lse
            final_loss = loss
        if process_group is not None:
            reduced_loss = all_reduce(final_loss, process_group=process_group)
            reduced_lse = all_reduce(final_lse, process_group=process_group)
        else:
            reduced_loss = reduce_scatter(final_loss, process_group=process_group)
            reduced_lse = reduce_scatter(final_lse, process_group=process_group)
        reduced_loss *= logit_scale * logit_scale
        reduced_lse *= logit_scale * logit_scale
        reduced_lse *= 0.5
        ctx.mark_non_differentiable(reduced_lse)
        return reduced_loss + reduced_lse, reduced_lse


def cross_entropy_loss(
    logits,  # logits is always required
    labels,  # labels is always required
    smoothing=0.0,
    logit_scale=1.0,
    lse_square_scale=0.0,
    ignored_index=-100,
    process_group=None,
):
    if isinstance(labels, tuple) and len(labels) == 2:
        labels, global_labels = labels
        assert global_labels is not None
    else:
        global_labels = None
    ret, _ = CrossEntropyLoss.apply(
        logits,
        labels,
        smoothing,
        logit_scale,
        lse_square_scale,
        ignored_index,
        process_group,
    )
    return ret
