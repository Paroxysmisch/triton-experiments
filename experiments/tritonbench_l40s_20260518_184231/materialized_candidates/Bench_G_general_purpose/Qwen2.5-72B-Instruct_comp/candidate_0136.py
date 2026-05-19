import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,  # Pointer to the logits tensor
    labels_ptr,  # Pointer to the labels tensor
    loss_ptr,    # Pointer to the loss tensor
    lse_ptr,     # Pointer to the log-sum-exp tensor
    num_classes, # Number of classes
    smoothing,   # Label smoothing factor
    scale,       # Scaling factor for logits
    ignore_index, # Label to ignore
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = min(batch_start + BLOCK_SIZE, logits_ptr.shape[0])

    for i in range(batch_start, batch_end):
        row = logits_ptr[i, :]
        row_scaled = row * scale
        max_logit = tl.max(row_scaled, axis=0)
        row_scaled -= max_logit
        exp_row = tl.exp(row_scaled)
        sum_exp = tl.sum(exp_row, axis=0)
        lse = max_logit + tl.log(sum_exp)
        label = labels_ptr[i]

        if label == ignore_index:
            loss_ptr[i] = 0.0
        else:
            if label < num_classes:
                if smoothing > 0.0:
                    loss_ptr[i] = lse - (row_scaled[label] * (1.0 - smoothing) + tl.log(num_classes) * smoothing)
                else:
                    loss_ptr[i] = lse - row_scaled[label]
            else:
                loss_ptr[i] = 0.0

        lse_ptr[i] = lse

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,  # Pointer to the logits tensor
    labels_ptr,  # Pointer to the labels tensor
    lse_ptr,     # Pointer to the log-sum-exp tensor
    grad_ptr,    # Pointer to the gradient tensor
    num_classes, # Number of classes
    smoothing,   # Label smoothing factor
    scale,       # Scaling factor for logits
    ignore_index, # Label to ignore
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = min(batch_start + BLOCK_SIZE, logits_ptr.shape[0])

    for i in range(batch_start, batch_end):
        row = logits_ptr[i, :]
        row_scaled = row * scale
        max_logit = tl.max(row_scaled, axis=0)
        row_scaled -= max_logit
        exp_row = tl.exp(row_scaled)
        sum_exp = tl.sum(exp_row, axis=0)
        lse = lse_ptr[i]
        label = labels_ptr[i]

        if label == ignore_index:
            grad_ptr[i, :] = 0.0
        else:
            if label < num_classes:
                prob = exp_row / sum_exp
                if smoothing > 0.0:
                    prob -= (1.0 - smoothing) * (tl.arange(0, num_classes) == label) + smoothing / num_classes
                else:
                    prob -= (tl.arange(0, num_classes) == label)
                grad_ptr[i, :] = prob * scale
            else:
                grad_ptr[i, :] = 0.0

import torch

class CrossEntropyLoss:
    def __init__(self, num_classes, smoothing=0.0, scale=1.0, ignore_index=-100):
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.scale = scale
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        batch_size = logits.shape[0]
        loss = torch.zeros(batch_size, device=logits.device)
        lse = torch.zeros(batch_size, device=logits.device)

        grid = (batch_size // 128 + 1,)
        cross_entropy_fwd_kernel[grid](logits, labels, loss, lse, self.num_classes, self.smoothing, self.scale, self.ignore_index, BLOCK_SIZE=128)

        return loss, lse

    def backward(self, logits, labels, grad_output, grad_logits):
        batch_size = logits.shape[0]
        grad = torch.zeros_like(logits)

        grid = (batch_size // 128 + 1,)
        cross_entropy_bwd_kernel[grid](logits, labels, lse, grad, self.num_classes, self.smoothing, self.scale, self.ignore_index, BLOCK_SIZE=128)

        grad_logits.add_(grad * grad_output.unsqueeze(1))

def cross_entropy_loss(logits, labels, num_classes, smoothing=0.0, scale=1.0, ignore_index=-100, in_place_backward=False):
    criterion = CrossEntropyLoss(num_classes, smoothing, scale, ignore_index)
    loss, lse = criterion.forward(logits, labels)

    if in_place_backward:
        grad_logits = logits
    else:
        grad_logits = torch.zeros_like(logits)

    grad_output = torch.ones_like(loss)
    criterion.backward(logits, labels, grad_output, grad_logits)

    return loss, lse, grad_logits

import torch

# Example data
logits = torch.randn(10, 100, device='cuda')
labels = torch.randint(0, 100, (10,), device='cuda')

# Compute loss and gradients
loss, lse, grad_logits = cross_entropy_loss(logits, labels, num_classes=100, smoothing=0.1, scale=1.0, ignore_index=-100, in_place_backward=False)

print("Loss:", loss)
print("Log-Sum-Exp:", lse)
print("Gradient of Logits:", grad_logits)
