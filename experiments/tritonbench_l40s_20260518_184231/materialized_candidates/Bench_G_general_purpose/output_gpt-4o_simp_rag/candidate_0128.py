import torch
import triton
import triton.language as tl

class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-1):
        self.smoothing = smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.ignored_index = ignored_index

    def forward(self, logits, labels, total_classes, class_start_idx=0, split=False):
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = 128  # Define block size for parallelism
        # Allocate memory for output tensors
        loss = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
        lse = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
        z_loss = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)

        # Launch Triton kernel
        grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_fwd_kernel[grid](
            loss,
            lse,
            z_loss,
            logits,
            labels,
            self.smoothing,
            self.logit_scale,
            self.lse_square_scale,
            self.ignored_index,
            total_classes,
            class_start_idx,
            n_cols,
            n_rows,
            logits.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=self.smoothing > 0.0,
            SPLIT=split,
        )
        return loss, lse

    def backward(self, dloss, logits, lse, labels, total_classes, class_start_idx=0):
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = 128  # Define block size for parallelism
        # Allocate memory for gradient tensor
        dlogits = torch.empty_like(logits)

        # Launch Triton kernel
        grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_bwd_kernel[grid](
            dlogits,
            dloss,
            logits,
            lse,
            labels,
            self.smoothing,
            self.logit_scale,
            self.lse_square_scale,
            self.ignored_index,
            total_classes,
            class_start_idx,
            n_cols,
            logits.stride(0),
            dlogits.stride(0),
            dloss.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=self.smoothing > 0.0,
        )
        return dlogits


def cross_entropy_loss(logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-1, total_classes=None, class_start_idx=0, split=False):
    if total_classes is None:
        total_classes = logits.size(1)
    
    # Instantiate the loss class
    loss_fn = CrossEntropyLoss(smoothing, logit_scale, lse_square_scale, ignored_index)
    
    # Compute forward pass
    loss, lse = loss_fn.forward(logits, labels, total_classes, class_start_idx, split)
    
    # Compute backward pass
    # Note: This example assumes dloss is available for backward pass
    dloss = torch.ones_like(loss)  # Example: gradient of loss w.r.t. itself
    dlogits = loss_fn.backward(dloss, logits, lse, labels, total_classes, class_start_idx)
    
    return loss, dlogits
