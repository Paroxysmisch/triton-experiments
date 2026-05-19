import triton
import triton.language as tl
import torch

@triton.jit
def _fused_cross_entropy_log_softmax_kernel(
    input_ptr, target_ptr, output_ptr,
    N, C,
    stride_n, stride_c,
    ignore_index,
    label_smoothing,
    use_weight,
    weight_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    # Each block processes one row (batch element)
    # If pid_n >= N, just return
    if pid_n >= N:
        return

    # Pointers to the row we are processing
    row_input_ptr = input_ptr + pid_n * stride_n
    row_target = tl.load(target_ptr + pid_n)
    # Skip row if ignore_index is set and matched
    skip_row = (row_target == ignore_index)

    # 1) Compute row max for numerical stability
    # We'll fetch in increments of BLOCK_SIZE
    idxs = tl.arange(0, BLOCK_SIZE)
    max_val = tl.float32(-1e30)
    # We'll loop over columns with step BLOCK_SIZE
    for start_c in range(0, C, BLOCK_SIZE):
        mask = idxs + start_c < C
        row_val = tl.where(
            mask,
            tl.load(row_input_ptr + (start_c + idxs) * stride_c),
            # Large negative for masked out
            tl.float32(-1e30)
        )
        block_max = tl.maximum(tl.max(row_val, axis=0), max_val)
        max_val = block_max
    # Broadcast final max to all threads in block
    row_max = tl.max(max_val, axis=0)

    # 2) Compute sum of exp(values - row_max) and optional sum of input for label smoothing
    sum_exp = tl.float32(0.)
    sum_input = tl.float32(0.)
    for start_c in range(0, C, BLOCK_SIZE):
        mask = idxs + start_c < C
        row_val = tl.where(
            mask,
            tl.load(row_input_ptr + (start_c + idxs) * stride_c),
            0.0
        )
        shifted = row_val - row_max
        exp_val = tl.exp(shifted)
        sum_exp += tl.sum(tl.where(mask, exp_val, 0.), axis=0)
        sum_input += tl.sum(tl.where(mask, row_val, 0.), axis=0)

    log_sum_exp = tl.log(sum_exp)

    # 3) Compute the cross entropy for this row
    # We'll do partial computations for label smoothing if not skipping
    loss_val = tl.float32(0.)
    if skip_row:
        # If target is to be ignored, set loss to 0
        loss_val = 0.
    else:
        # Confirm valid target in [0, C)
        # If it's out of range, we can set 0 as well
        row_target = tl.where(
            (row_target >= 0) & (row_target < C),
            row_target,
            -1
        )
        # We'll find log_smax for the correct class and also sum log_smax if label_smoothing
        correct_col = row_target.to(tl.int32)
        # We'll do a second pass to gather log_smax_correct and sum_log_smax
        sum_log_smax = tl.float32(0.)
        log_smax_correct = tl.float32(0.)

        running_correct = tl.float32(0.)
        running_sum_log_smax = tl.float32(0.)

        for start_c in range(0, C, BLOCK_SIZE):
            mask = idxs + start_c < C
            row_val = tl.where(
                mask,
                tl.load(row_input_ptr + (start_c + idxs) * stride_c),
                0.0
            )
            shifted = row_val - row_max
            log_smax = shifted - log_sum_exp
            # Accumulate sum_log_smax
            running_sum_log_smax += tl.sum(tl.where(mask, log_smax, 0.), axis=0)

            # If correct_col is within this block, fetch that log_smax
            cond_correct = (correct_col == (start_c + idxs))
            log_smax_c = tl.where(cond_correct, log_smax, 0.)
            running_correct += tl.sum(log_smax_c, axis=0)

        sum_log_smax = running_sum_log_smax
        log_smax_correct = running_correct

        # label_smoothing formula:
        # - [ (1 - ls) * log_smax_correct + (ls / (C-1)) * (sum_log_smax - log_smax_correct ) ]
        ls = label_smoothing
        if row_target == -1:  # invalid target
            loss_val = 0.
        else:
            n_others = C - 1
            base_loss = -(1.0 - ls) * log_smax_correct
            if ls > 0.0 and n_others > 0:
                base_loss += -(ls / n_others) * (sum_log_smax - log_smax_correct)
            # optional weighting for correct class
            if use_weight != 0:
                class_wt = tl.load(weight_ptr + row_target) if row_target >= 0 else 1.0
                base_loss *= class_wt
            loss_val = base_loss

    # Store per-sample loss
    tl.store(output_ptr + pid_n, loss_val)


def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Fused cross entropy + log softmax in Triton
    """
    # For simplicity, we only handle dim=1 in this example.
    # If dim != 1, we can transpose or raise error.
    if dim != 1:
        raise NotImplementedError("This fused Triton kernel currently only supports dim=1.")
    if not input.is_cuda or not target.is_cuda:
        raise ValueError("Input and target must both be CUDA tensors for Triton kernel.")

    # Check shapes
    N, C = input.shape[0], input.shape[1]
    if target.shape[0] != N:
        raise ValueError("Target shape must match input batch size along dim=0.")

    # Prepare output tensor for per-sample losses
    loss_out = torch.empty((N,), dtype=input.dtype, device=input.device)

    # Kernel grid
    grid = (N,)

    # Whether to use class weighting
    use_weight = 1 if weight is not None else 0
    if weight is None:
        weight_ptr = torch.zeros((1,), device=input.device, dtype=input.dtype)
    else:
        if weight.numel() != C:
            raise ValueError("Weight tensor must have number of elements equal to C (classes).")
        weight_ptr = weight

    BLOCK_SIZE = 1024  # tune if needed
    # Launch Triton kernel
    _fused_cross_entropy_log_softmax_kernel[grid](
        input_ptr=input.data_ptr(),
        target_ptr=target.data_ptr(),
        output_ptr=loss_out.data_ptr(),
        N=N,
        C=C,
        stride_n=input.stride(0),
        stride_c=input.stride(1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        use_weight=use_weight,
        weight_ptr=weight_ptr.data_ptr(),
        BLOCK_SIZE=BLOCK_SIZE
    )

    if reduction == 'none':
        return loss_out
    elif reduction == 'mean':
        return loss_out.mean()
    elif reduction == 'sum':
        return loss_out.sum()
    else:
        raise ValueError("reduction must be one of: 'none', 'mean', 'sum'.")
