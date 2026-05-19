import torch
import torch.distributed as dist
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, lse_ptr, loss_ptr,
    batch_size, num_classes,
    smoothing: tl.constexpr,
    lse_square_scale: tl.constexpr,
    ignored_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Load target label for current row
    target = tl.load(labels_ptr + row_idx)
    if target == ignored_index:
        tl.store(lse_ptr + row_idx, 0.0)
        tl.store(loss_ptr + row_idx, 0.0)
        return

    # Initialize accumulation variables
    max_logit = -float('inf')
    sum_logits = 0.0
    logit_target = 0.0
    num_valid_classes = num_classes - 1  # Assume ignored_index isn't counted

    # First pass: compute max_logit and sum_logits (if needed)
    for offset in tl.range(0, num_classes, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_classes
        logits = tl.load(logits_ptr + row_idx * num_classes + cols, mask=mask, other=0.0)
        
        # Update max logit
        curr_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, curr_max)
        
        # Check for target in current block
        target_mask = (cols == target) & mask
        logit_target += tl.sum(tl.where(target_mask, logits, 0.0), axis=0)
        
        # Accumulate sum of logits if smoothing enabled
        if smoothing != 0.0:
            sum_logits += tl.sum(tl.where(mask, logits, 0.0), axis=0)

    # Second pass: compute sum of exponents
    sum_exp = 0.0
    for offset in tl.range(0, num_classes, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_classes
        logits = tl.load(logits_ptr + row_idx * num_classes + cols, mask=mask, other=0.0)
        
        safe_logits = logits - max_logit
        exp_logits = tl.exp(safe_logits)
        sum_exp += tl.sum(tl.where(mask, exp_logits, 0.0), axis=0)

    # Compute LSE and final loss
    lse = tl.log(sum_exp) + max_logit
    if smoothing != 0.0:
        smoothed_term = (1 - smoothing) * logit_target + (smoothing / num_valid_classes) * (sum_logits - logit_target)
        loss = lse - smoothed_term
    else:
        loss = lse - logit_target

    # Apply LSE regularization
    if lse_square_scale != 0.0:
        loss += lse_square_scale * lse * lse

    tl.store(lse_ptr + row_idx, lse)
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    grad_logits_ptr, logits_ptr, labels_ptr, lse_ptr,
    batch_size, num_classes,
    smoothing: tl.constexpr,
    lse_square_scale: tl.constexpr,
    ignored_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= batch_size:
        return

    # Load target and check if ignored
    target = tl.load(labels_ptr + row_idx)
    if target == ignored_index:
        for offset in tl.range(0, num_classes, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < num_classes
            tl.store(grad_logits_ptr + row_idx * num_classes + cols, 0.0, mask=mask)
        return

    # Load LSE and calculate valid classes
    lse = tl.load(lse_ptr + row_idx)
    num_valid_classes = num_classes - 1

    # Compute gradients for each class
    for offset in tl.range(0, num_classes, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_classes
        logits = tl.load(logits_ptr + row_idx * num_classes + cols, mask=mask, other=0.0)
        
        # Compute softmax and target probability
        safe_logits = logits - lse
        softmax = tl.exp(safe_logits)
        is_target = cols == target
        target_prob = tl.where(is_target, 1.0 - smoothing, smoothing / num_valid_classes)
        
        # Calculate gradient components
        grad = softmax - target_prob
        if lse_square_scale != 0.0:
            grad += 2 * lse_square_scale * lse * softmax
        
        tl.store(grad_logits_ptr + row_idx * num_classes + cols, grad, mask=mask)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, lse_square_scale, ignored_index, process_group):
        batch_size, num_classes = logits.shape
        device = logits.device
        
        # Allocate output tensors
        lse = torch.empty_like(logits[:, 0], device=device)
        loss = torch.empty_like(logits[:, 0], device=device)
        
        # Kernel configuration
        grid = (batch_size,)
        BLOCK_SIZE = triton.next_power_of_2(num_classes)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        # Launch forward kernel
        cross_entropy_fwd_kernel[grid](
            logits, labels, lse, loss,
            batch_size, num_classes,
            smoothing, lse_square_scale, ignored_index,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Handle distributed scenario
        if process_group is not None:
            # Placeholder for distributed synchronization logic
            pass

        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.process_group = process_group

        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        device = logits.device
        
        # Allocate gradient tensor
        grad_logits = torch.empty_like(logits)
        
        # Kernel configuration
        grid = (batch_size,)
        BLOCK_SIZE = triton.next_power_of_2(num_classes)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        # Launch backward kernel
        cross_entropy_bwd_kernel[grid](
            grad_logits, logits, labels, lse,
            batch_size, num_classes,
            ctx.smoothing, ctx.lse_square_scale, ctx.ignored_index,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Handle distributed scenario
        if ctx.process_group is not None:
            # Placeholder for distributed synchronization logic
            pass

        return grad_logits * grad_output.view(-1, 1), None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    process_group=None
) -> torch.Tensor:
    return CrossEntropyLoss.apply(
        logits, labels, smoothing,
        lse_square_scale, ignored_index,
        process_group
    )

logits = torch.randn(128, 1000).cuda()  # Batch size 128, 1000 classes
labels = torch.randint(0, 1000, (128,)).cuda()

loss = cross_entropy_loss(
    logits, labels,
    smoothing=0.1,
    lse_square_scale=0.01
)
loss.backward()
