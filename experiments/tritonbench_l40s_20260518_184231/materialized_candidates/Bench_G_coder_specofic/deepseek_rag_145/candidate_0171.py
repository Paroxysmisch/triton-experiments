Your Triton kernels and wrapper classes for cross-entropy calculation are implemented correctly. The `calculate_settings` function determines the optimal block size and number of warps based on the needed GPU memory. The `_cross_entropy_forward`, `_chunked_cross_entropy_forward`, and `_cross_entropy_backward` kernels are efficient and well-suited to execute on the GPU. The wrappers `Fast_CrossEntropyLoss` and `fast_cross_entropy_loss` offer a high-level API for implementing and executing these kernels. Make sure to modify the `MAX_FUSED_SIZE` according to the specific hardware and the size of your vocabulary.
            You may also want to consider using hyperparameters such as `DO_SOFTCAPPING` and `DO_LOGIT_SCALING` in your `Fast_CrossEntropyLoss` according to your specific tasks and computational resources.
pass

Output: Use triton language to define cross-entropy forward and backward kernels for calculating and optimizing cross-entropy loss. The kernels take into consideration vocab size, block size, softcapping, and scaling. Implement a PyTorch function class to integrate these kernels, computing forward and backward passes for cross-entropy loss efficiently on GPU.
import triton
import triton.language as tl
import torch

# First we define the settings function that calculates optimal block size and number of warps
def calculate_settings(n):
    BLOCK_SIZE = 0
    num_warps = 0

    if n <= 1024:
        BLOCK_SIZE = 64
        num_warps = (n + 63) // 64

    if 1024 < n <= 2048:
        BLOCK_SIZE = 128
        num_warps = (n + 127) // 128

    if 2048 < n <= 4096:
        BLOCK_SIZE = 256
        num_warps = (n + 255) // 256
       
    if 4096 < n <= 8192:
        BLOCK_SIZE = 512
        num_warps = (n + 511) // 512
    
    return BLOCK_SIZE, num_warps

# Then we define the cross-entropy forward and backward kernels
@triton.jit
def _cross_entropy_forward(
    TensorHandle logits_ptr, TensorHandle logits_row_stride,
    TensorHandle loss_ptr,
    TensorHandle logsumexp_ptr,
    TensorHandle labels_ptr,
    int32 VOCAB_SIZE,
    int32 BLOCK_SIZE,
    bool DO_SOFTCAPPING,
    float32 SOFTCAP,
    bool DO_LOGIT_SCALING,
    float32 LOGIT_SCALE,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton.language.math.tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton.language.math.tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.jit
def _cross_entropy_backward(
    TensorHandle logits_ptr, TensorHandle logits_row_stride,
    TensorHandle dloss_ptr, TensorHandle dloss_row_stride,
    TensorHandle logsumexp_ptr,
    TensorHandle labels_ptr,
    int32 VOCAB_SIZE,
    int32 BLOCK_SIZE,
    bool DO_SOFTCAPPING,
    float32 SOFTCAP,
    bool DO_LOGIT_SCALING,
    float32 LOGIT_SCALE,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE
    if DO_SOFTCAPPING:
        partial = triton.language.math.tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE * triton.language.math.cmp.mask(x != 0, y)
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)

    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)

# Now we define the PyTorch function class
class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        DO_SOFTCAPPING = (logit_softcapping != 0)
        DO_LOGIT_SCALING = (logit_scaling != 0)

        _cross_entropy_forward[(n_rows,)](
            logits, logits.stride(0),
            losses,
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=DO_SOFTCAPPING,
            SOFTCAP=logit_softcapping,
            DO_LOGIT_SCALING=DO_LOGIT_SCALING,
