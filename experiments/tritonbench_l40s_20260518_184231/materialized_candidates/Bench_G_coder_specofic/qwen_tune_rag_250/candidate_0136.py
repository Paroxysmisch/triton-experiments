lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    if SPLIT:
        tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
        lse = tl.load(lse_ptr + col_block_idx * n_rows + row_idx)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        if label_idx >= class_start_idx and label_idx < min(n_cols, class_start_idx + BLOCK_SIZE):
            correct_logit = tl.load(logits_ptr + label_idx - class_start_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse - correct_logit if label_idx < n_cols else 0.0)
                    + smoothing * (sum_logits - correct_logit) / total_classes
                )
            else:
                loss = lse - correct_logit if label_idx < n_cols else 0.0
        else:
            # If the label is out of the range of classes handled by this tensor, we still want the loss to be 0.
            # This handles the case where the dataset across ranks is not perfectly divisible by the number of ranks.
            loss = 0.0
        if HAS_SMOOTHING:
            loss += (1 - smoothing) * tl.where(label_idx < n_cols, -1.0 / total_classes, 0.0)
        z_loss = lse_square_scale * lse * lse
    if row_idx % 1024 == 0:
        print(f"Rank {process_group.get_global_rank()} row {row_idx} of {n_rows}")
    tl.store(loss_ptr + row_idx, loss)
    if z_loss_ptr is not None:
        tl.store(z_loss_ptr + row_idx, z_loss)


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
    n_rows,
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
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - smooth_positive, probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)


class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignored_index=-100,
        inplace_backward=False,
    ):
        if smoothing < 0 or smoothing > 1:
            raise ValueError(f"Smoothing proportion must be between 0 and 1, but got {smoothing}")
        if logit_scale <= 0:
            raise ValueError(f"Logit scale must be strictly positive, but got {logit_scale}")
        if not logits.is_contiguous() or not labels.is_contiguous():
            raise ValueError("Both logits and labels must be contiguous")
        if logits.ndim != 2:
            raise ValueError(f"Logits must be 2D, but got {logits.ndim}")
        if labels.ndim != 1:
            raise ValueError(f"Labels must be 1D, but got {labels.ndim}")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError(
                "Dimension mismatch between logits and labels (expected logits[1] == labels[0])"
            )
        total_classes = logits.shape[1]
        n_rows, n_cols = logits.shape
        process_group = ProcessGroup.get_default_process_group()
        world_size = process_group.get_world_size()
        rank = process_group.get_global_rank()
        # This is useful when you have tensor parallel, and each rank only has a subset of the classes to handle
        # E.g., world_size == 2, rank == 0 only has classes 0, 2, 4, ... and rank == 1 only has classes 1, 3, 5, ...
        class_start_idx = rank * (total_classes // world_size)
        # The case when total_classes is not perfectly divisible by world_size is handled gracefully by the recipient
        # That is, it just doesn't try to process the extra classes
        logits_row_stride = logits.stride(0)
        dlogits = torch.empty_like(logits, dtype=torch.float32, requires_grad=inplace_backward)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        z_losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse_square_scale_nonzero = lse_square_scale != 0.0
        lse_ptr = (
            torch.empty(
                world_size, n_rows, dtype=torch.float32, device=logits.device
            ) if lse_square_scale_nonzero
            else None
        )
        z_loss_ptr = (
            torch.empty(n_rows, dtype=torch.float32, device=logits.device)
            if lse_square_scale_nonzero
            else None
        )
        dlogits_row_stride = dlogits.stride(0)
        dlosses = torch.ones(n_rows, dtype=torch.float32, device=logits.device)
        dloss_ptr = (
            dlosses if not inplace_backward else torch.empty_like(losses, dtype=torch.float32)
        )
        dloss_row_stride = dloss_ptr.stride(0)
        has_smoothing = smoothing > 0.0
        BLOCK_SIZE = max(16, next_power_of_2(total_classes // world_size))
        grid = lambda meta: (n_rows, triton.cdiv(total_classes, meta["BLOCK_SIZE"]))
        cross_entropy_fwd_kernel[grid](
            losses,  # data ptrs
            lse_ptr,
            z_loss_ptr,
            logits,
            labels,
            smoothing,  # scalars
            logit_scale,
            lse_square_scale,
            ignored_index,
            total_classes,
            class_start_idx,
            n_cols,  # shapes
            n_rows,
            logits_row_stride,  # strides
            BLOCK_SIZE=BLOCK_SIZE,  # constants
            HAS_SMOOTHING=has_smoothing,
            SPLIT=lse_square_scale_nonzero,
            num_warps=4,
        )
        if lse_square_scale_nonzero:
            # Only do this reduction if we actually need the z-loss
            z_losses = torch.sum(torch.exp(2.0 * lse_ptr) * losses[:, None], axis=0)
        if rank == 0:
            # This will be zero if the dataset across ranks is not perfectly divisible by the number of ranks
            n_rows_per_rank = n_rows // world_size
            # We keep the losses for all ranks contiguous in memory. This is a waste of memory for the non-zero entries,
            # but avoids a lot of complicated indexing
            losses = losses.view(world_size, n_rows_per_rank)
            z_losses = z_losses.view(world_size, n_rows_per_rank) if lse_square_scale_nonzero else None
            dlosses = dlosses.view(world_size, n_rows_per_rank)
            # Note that we skip the rows corresponding to the extra classes handled by other ranks
            for i in range(1, world_size):
                rank_losses = torch.empty(n_rows_per_rank, dtype=torch.float32, device=logits.device)
                if lse_square_scale_nonzero:
                    rank_z_losses = torch.empty(n_rows_per_rank, dtype=torch.float32, device=logits.device)
                else:
                    rank_z_losses = None
                rank_dlosses = torch.empty(n_rows_per_rank, dtype=torch.float32, device=logits.device)
                process_group.recv(rank_losses, src=i)
                if lse_square_scale_nonzero:
                    process_group.recv(rank_z_losses, src=i)
                process_group.recv(rank_dlosses, src=i)
                losses[i] += rank_losses
                if lse_square_scale_nonzero:
                    z_losses[i] += rank_z_losses
