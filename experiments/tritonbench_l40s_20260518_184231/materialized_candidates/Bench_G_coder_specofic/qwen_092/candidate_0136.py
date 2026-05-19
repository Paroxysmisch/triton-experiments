import triton
import triton.language as tl

# Kernel for forward pass of cross-entropy loss
@triton.jit
def cross_entropy_fwd_kernel(
    logits, labels, lse, loss, labels_ignore, smoothing, logit_scale, class_offset, n_classes, stride, n_row, block_size, n_warps
):
    pid = tl.program_id(axis=0)
    row_start = pid * block_size
    row_end = min(row_start + block_size, n_row)
    row_range = row_end - row_start

    labels = labels[row_start:row_end]
    logits = logits[row_start * stride:(row_end * stride) + n_classes]
    lse[row_start:row_end] = 0.0
    loss[row_start:row_end] = 0.0

    for row in range(row_range):
        row_labels = labels[row]
        row_logits = logits[row * stride:(row + 1) * stride]
        row_lse = 0.0
        row_loss = 0.0

        for i in range(n_classes):
            if row_labels[i] != labels_ignore:
                logit = row_logits[i] * logit_scale
                logit -= class_offset
                logit = tl.where(logit < 0, logit, -tl.log(tl.exp(-logit)))
                row_lse += logit
                if row_labels[i] == 0:
                    row_loss += logit
                else:
                    row_loss += 0.0

        row_lse = tl.math.logsumexp(row_logits * logit_scale - class_offset)
        lse[row] = row_lse
        loss[row] = row_loss

    if smoothing > 0.0:
        for row in range(row_range):
            row_labels = labels[row]
            row_loss = loss[row]
            for i in range(n_classes):
                if row_labels[i] != labels_ignore:
                    row_loss += smoothing * (tl.math.logsumexp(row_logits * logit_scale - class_offset) - row_lse)
            loss[row] = row_loss

# Kernel for backward pass of cross-entropy loss
@triton.jit
def cross_entropy_bwd_kernel(
    logits, labels, lse, loss, gradients, labels_ignore, smoothing, logit_scale, class_offset, n_classes, stride, n_row, block_size, n_warps
):
    pid = tl.program_id(axis=0)
    row_start = pid * block_size
    row_end = min(row_start + block_size, n_row)
    row_range = row_end - row_start

    labels = labels[row_start:row_end]
    logits = logits[row_start * stride:(row_end * stride) + n_classes]
    lse[row_start:row_end] = 0.0
    loss[row_start:row_end] = 0.0
    gradients[row_start * stride:(row_end * stride) + n_classes] = 0.0

    for row in range(row_range):
        row_labels = labels[row]
        row_logits = logits[row * stride:(row + 1) * stride]
        row_lse = 0.0
        row_loss = 0.0

        for i in range(n_classes):
            if row_labels[i] != labels_ignore:
                logit = row_logits[i] * logit_scale
                logit -= class_offset
                logit = tl.where(logit < 0, logit, -tl.log(tl.exp(-logit)))
                row_lse += logit
                if row_labels[i] == 0:
                    row_loss += logit
                else:
                    row_loss += 0.0

        row_lse = tl.math.logsumexp(row_logits * logit_scale - class_offset)
        lse[row] = row_lse
        loss[row] = row_loss

    if smoothing > 0.0:
        for row in range(row_range):
            row_labels = labels[row]
            row_loss = loss[row]
            for i in range(n_classes):
                if row_labels[i] != labels_ignore:
                    row_loss += smoothing * (tl.math.logsumexp(row_logits * logit_scale - class_offset) - row_lse)
            loss[row] = row_loss

    for row in range(row_range):
        row_labels = labels[row]
        row_logits = logits[row * stride:(row + 1) * stride]
        row_lse = 0.0
        row_loss = 0.0

        for i in range(n_classes):
            if row_labels[i] != labels_ignore:
                logit = row_logits[i] * logit_scale
                logit -= class_offset
                logit = tl.where(logit < 0, logit, -tl.log(tl.exp(-logit)))
                row_lse += logit
                if row_labels[i] == 0:
                    row_loss += logit
                else:
                    row_loss += 0.0

        row_lse = tl.math.logsumexp(row_logits * logit_scale - class_offset)
        lse[row] = row_lse
        loss[row] = row_loss

    if smoothing > 0.0:
        for row in range(row_range):
            row_labels = labels[row]
            row_loss = loss[row]
            for i in range(n_classes):
                if row_labels[i] != labels_ignore:
                    row_loss += smoothing * (tl.math.logsumexp(row_logits * logit_scale - class_offset) - row_lse)
            loss[row] = row_loss

# Wrapper class for cross-entropy loss
class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, logit_scale=1.0, class_offset=0.0):
        self.smoothing = smoothing
        self.logit_scale = logit_scale
        self.class_offset = class_offset

    def forward(self, logits, labels, labels_ignore=-1):
        n_classes = logits.shape[1]
        n_row = logits.shape[0]
        stride = n_classes
        block_size = 32
        n_warps = 4

        lse = tl.zeros(n_row, dtype=tl.float32)
        loss = tl.zeros(n_row, dtype=tl.float32)

        cross_entropy_fwd_kernel[grid=(n_row // block_size), block=(block_size, 1, 1)](
            logits, labels, lse, loss, labels_ignore, self.smoothing, self.logit_scale, self.class_offset, n_classes, stride, n_row, block_size, n_warps
        )

        return loss, lse

    def backward(self, logits, labels, lse, loss, gradients, labels_ignore=-1):
        n_classes = logits.shape[1]
        n_row = logits.shape[0]
        stride = n_classes
        block_size = 32
        n_warps = 4

        cross_entropy_bwd_kernel[grid=(n_row // block_size), block=(block_size, 1, 1)](
            logits, labels, lse, loss, gradients, labels_ignore, self.smoothing, self.logit_scale, self.class_offset, n_classes, stride, n_row, block_size, n_warps
        )

        return gradients

# High-level entry point for cross-entropy loss
def cross_entropy_loss(logits, labels, smoothing=0.0, logit_scale=1.0, class_offset=0.0, labels_ignore=-1):
    n_classes = logits.shape[1]
    n_row = logits.shape[0]
    stride = n_classes

    loss = tl.zeros(n_row, dtype=tl.float32)
    lse = tl.zeros(n_row, dtype=tl.float32)
    gradients = tl.zeros((n_row * stride), dtype=tl.float32)

    cross_entropy_loss_obj = CrossEntropyLoss(smoothing, logit_scale, class_offset)
    loss, lse = cross_entropy_loss_obj.forward(logits, labels, labels_ignore)
    gradients = cross_entropy_loss_obj.backward(logits, labels, lse, loss, gradients, labels_ignore)

    return loss, lse, gradients
