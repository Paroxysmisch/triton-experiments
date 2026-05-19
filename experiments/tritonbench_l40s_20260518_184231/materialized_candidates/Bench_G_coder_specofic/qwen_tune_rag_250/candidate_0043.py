mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        # We compute the gradient of y, not y in the log-space
        if not log_target:
            new_grad = target - 1.0
        else:
            new_grad = tl.exp(target) - 1.0

        tl.store(new_grads_ptr + offsets, new_grad, mask=mask)


def kldiv_forward_triton(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool,
    reduction: str,
    eps: float = 1e-8,
):
    """
    Compute the Kullback-Leibler divergence between the prediction and the ground truth.
    The prediction should always be in the log-space.
    This function is Triton implementation.
    """
    if y_pred.stride(-1) != 1:
        y_pred = y_pred.contiguous()
    if y_true.stride(-1) != 1:
        y_true = y_true.contiguous()

    n_cols = y_pred.shape[-1]
    assert y_true.shape[-1] == n_cols

    assert reduction in (
        _REDUCTION_MODE_NONE,
        _REDUCTION_MODE_SUM,
        _REDUCTION_MODE_MEAN,
        _REDUCTION_MODE_BATCHMEAN,
    )

    if reduction == _REDUCTION_MODE_NONE:
        n_rows = y_pred.numel() // n_cols
        loss = torch.empty(n_rows, device=y_pred.device, dtype=torch.float32)
    else:
        loss = torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)

    # Triton kernel requires the input to be in the log-space
    _kldiv_kernel_forward[(y_pred.shape[0],)](
        y_pred,
        y_pred.stride(0),
        y_true,
        y_true.stride(0),
        loss,
        1 if reduction == _REDUCTION_MODE_NONE else 0,
        n_cols,
        eps,
        BLOCK_SIZE=n_cols,
        num_warps=1,
        log_target=log_target,
        reduction=reduction,
    )

    return loss


def kldiv_backward_triton(
    target: torch.Tensor,
    grad_output: torch.Tensor,
    new_grads: torch.Tensor,
    log_target: bool,
):
    """
    Compute the gradients of the Kullback-Leibler divergence.
    This function is Triton implementation.
    """
    assert grad_output.shape[-1] == 1

    n_cols = target.shape[-1]
    assert new_grads.shape[-1] == n_cols

    if new_grads.stride(-1) != 1:
        new_grads = new_grads.contiguous()

    _kldiv_kernel_backward[(target.shape[0],)](
        target,
        target.stride(0),
        new_grads,
        new_grads.stride(0),
        n_cols,
        BLOCK_SIZE=n_cols,
        num_warps=1,
        log_target=log_target,
    )
