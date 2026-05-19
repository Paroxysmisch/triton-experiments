import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    lse,
    loss,
    logits,
    logit_scale,
    label_smoothing,
    total_classes,
    classes_per_block,
    ignore_index,
    blocks_per_example,
    logits_row_stride,
    label_smoothing_factor: tl.constexpr,
    ignore_label_as_large_as: tl.constexpr,
    logit_scaling_importance: tl.constexpr,
    handle_specific_class: tl.constexpr,
    specific_class: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index and class index
    example_idx = tl.program_id(0)
    class_idx = tl.program_id(1)

    # Calculate the starting row index for the current block
    row_start_idx = example_idx * logits_row_stride
    row_idx_block_start = row_start_idx + class_idx * BLOCK_SIZE

    # Load logits and apply logit scaling
    logits_ptr = logits + row_idx_block_start
    lse_ptr = lse + row_idx_block_start

    # Handle specific class case
    if handle_specific_class:
        if specific_class != class_idx:
            tl.debug_barrier()
            return

    # Load logits and apply logit scaling
    if logit_scale is not None:
        logit_scale_ptr = logit_scale + row_idx_block_start
        logits = (logits - tl.log(tl.to(BLOCK_SIZE, tl.float32))) * tl.exp(
            logit_scale_ptr
        )

    # Load logits
    logits = tl.load(logits_ptr, mask=class_idx < total_classes, other=-float("inf"))

    if logit_scaling_importance:
        logits = tl.log(1 + tl.exp(logits))

    # Apply label smoothing
    if label_smoothing is not None:
        logits = (1 - label_smoothing_factor) * logits + label_smoothing_factor * -100

    # Compute lse and loss
    lse = tl.max(logits, 0)
    loss = lse + tl.log(tl.sum(tl.exp(logits - lse), 0))

    # Handle ignore index
    if ignore_index >= 0:
        ignore_mask = class_idx == ignore_index
        loss = tl.where(ignore_mask, 0, loss)
        if ignore_label_as_large_as:
            loss = tl.where(ignore_mask, -float("inf"), loss)

    # Store lse and loss
    tl.store(lse_ptr, lse.to(tl.float16), mask=class_idx < total_classes)
    tl.store(loss + row_start_idx, loss.to(tl.float16), mask=class_idx < total_classes)


@triton.jit
def cross_entropy_bwd_kernel(
    grad_logits,
    lse,
    loss,
    logits,
    logit_scale,
    label_smoothing,
    total_classes,
    classes_per_block,
    ignore_index,
    blocks_per_example,
    logits_row_stride,
    label_smoothing_factor: tl.constexpr,
    ignore_label_as_large_as: tl.constexpr,
    logit_scaling_importance: tl.constexpr,
    handle_specific_class: tl.constexpr,
    specific_class: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index and class index
    example_idx = tl.program_id(0)
    class_idx = tl.program_id(1)

    # Calculate the starting row index for the current block
    row_start_idx = example_idx * logits_row_stride
    row_idx_block_start = row_start_idx + class_idx * BLOCK_SIZE

    # Load lse and loss
    lse_ptr = lse + row_idx_block_start
    loss_ptr = loss + row_start_idx
    lse = tl.load(lse_ptr, mask=class_idx < total_classes)
    loss = tl.load(loss_ptr, mask=class_idx < total_classes)

    # Handle specific class case
    if handle_specific_class:
        if specific_class != class_idx:
            tl.debug_barrier()
            return

    # Load logits and apply logit scaling
    grad_logits_ptr = grad_logits + row_idx_block_start
    logits_ptr = logits + row_idx_block_start
    if logit_scale is not None:
        logit_scale_ptr = logit_scale + row_idx_block_start
        logits = (logits - tl.log(tl.to(BLOCK_SIZE, tl.float32))) * tl.exp(
            logit_scale_ptr
        )
    logits = tl.load(
        logits_ptr, mask=class_idx < total_classes, other=-float("inf")
    ).to(tl.float32)

    if logit_scaling_importance:
        logits = tl.log(1 + tl.exp(logits))

    # Apply label smoothing
    if label_smoothing is not None:
        logits = (1 - label_smoothing_factor) * logits + label_smoothing_factor * -100

    # Compute gradients
    probs = tl.exp(logits - lse)
    if class_idx < total_classes:
        if (
            ignore_index >= 0
            and ignore_index == class_idx
            and ignore_label_as_large_as
        ):
            probs = tl.where(class_idx == class_idx, 0, probs)

        grad = probs - tl.to(class_idx == class_idx, tl.float32)

    # Store gradients
    tl.store(
        grad_logits_ptr, grad.to(tl.float16), mask=class_idx < total_classes
    )  # .contiguous()


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        logit_scale,
        label_smoothing,
        ignore_index=-100,
        logit_scaling_importance=True,
        handle_specific_class=False,
        specific_class=None,
    ):
        # Ensure logits and labels are on the same device
        if logits.device != labels.device:
            if labels.device == torch.device("cuda"):
                logits = logits.cuda()
            else:
                labels = labels.cuda()

        # Ensure logits is in float32
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)

        # Get the number of examples, classes per block, and total classes
        num_examples = logits.shape[0]
        classes_per_block = min(triton.next_power_of_2(logits.shape[1]), 1024)
        total_classes = logits.shape[1]

        # Calculate blocks per example and total blocks
        blocks_per_example = triton.cdiv(total_classes, classes_per_block)
        total_blocks = num_examples * blocks_per_example

        # Prepare data structures for loss and lse
        losses = torch.zeros((num_examples,), dtype=torch.float32, device=logits.device)
        lse = torch
