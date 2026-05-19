import triton
import triton.language as tl

# Define the block size and number of warps for optimal performance
BLOCK_SIZE = 128
NUM_WARPS = 4

# Forward kernel for KL divergence
@triton.jit
def _kldiv_kernel_forward(y_pred_ptr, y_true_ptr, loss_ptr, log_target, eps, reduction, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block and grid indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    y_pred = tl.load(y_pred_ptr + offsets, mask=offsets < n_elements, other=0.0)
    y_true = tl.load(y_true_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute KL divergence
    if log_target:
        loss = tl.exp(y_true) * (y_true - y_pred)
    else:
        loss = y_true * (tl.log(y_true + eps) - y_pred)
    
    # Store loss
    if reduction == "none":
        tl.store(loss_ptr + offsets, loss, mask=offsets < n_elements)
    else:
        # Reduction logic (sum, mean, batchmean)
        reduced_loss = tl.sum(loss, axis=0)
        if reduction == "mean":
            reduced_loss /= n_elements
        elif reduction == "batchmean":
            reduced_loss /= (n_elements / BLOCK_SIZE)
        
        # Store the reduced loss
        if tl.program_id(0) == 0:
            tl.atomic_add(loss_ptr, reduced_loss)

# Forward wrapper function
def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="none", eps=1e-9):
    # Allocate memory for the loss
    loss = torch.empty_like(y_pred)
    
    # Launch the kernel
    n_elements = y_pred.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _kldiv_kernel_forward[grid](y_pred, y_true, loss, log_target, eps, reduction, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return loss

# Backward kernel for KL divergence
@triton.jit
def _kldiv_kernel_backward(target_ptr, grad_output_ptr, new_grads_ptr, log_target, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block and grid indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    target = tl.load(target_ptr + offsets, mask=offsets < n_elements, other=0.0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute gradients
    if log_target:
        new_grads = tl.exp(target) * grad_output
    else:
        new_grads = target * grad_output
    
    # Store new gradients
    tl.store(new_grads_ptr + offsets, new_grads, mask=offsets < n_elements)

# Backward wrapper function
def kldiv_backward_triton(target, grad_output, log_target=False):
    # Allocate memory for the new gradients
    new_grads = torch.empty_like(target)
    
    # Launch the kernel
    n_elements = target.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _kldiv_kernel_backward[grid](target, grad_output, new_grads, log_target, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return new_grads
