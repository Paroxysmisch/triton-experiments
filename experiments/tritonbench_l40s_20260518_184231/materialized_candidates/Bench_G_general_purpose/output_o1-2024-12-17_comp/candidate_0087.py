import torch
import triton
import triton.language as tl

# -----------------------------------------------------------
# 1) Constants and reduction mode definitions
# -----------------------------------------------------------

MAX_FUSED_SIZE = 65536 // 4

_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

_str_to_reduction_mode = {
    'none': _REDUCTION_MODE_NONE,
    'sum': _REDUCTION_MODE_SUM,
    'mean': _REDUCTION_MODE_MEAN,
    'batchmean': _REDUCTION_MODE_BATCHMEAN,
}

# -----------------------------------------------------------
# 2) Warp count selector
# -----------------------------------------------------------

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 16384:
        return 16
    elif BLOCK_SIZE >= 8192:
        return 8
    else:
        return 4

# -----------------------------------------------------------
# 3) KLDivergence Forward Kernel
#    KL(y_true || y) = y_true * ( log(y_true) - log(y) )
# -----------------------------------------------------------

@triton.jit
def _kldiv_kernel_forward(
    y_ptr, gt_ptr, out_ptr,
    n_elements,
    stride_y, stride_gt, stride_out,
    reduction_mode, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load prediction and ground truth
    y_val = tl.load(y_ptr + offsets * stride_y, mask=mask, other=0.0)
    gt_val = tl.load(gt_ptr + offsets * stride_gt, mask=mask, other=0.0)

    # Compute KL divergence
    # To avoid log(0), add a small epsilon for numerical stability if desired
    eps = 1e-12
    out_val = gt_val * (tl.log(gt_val + eps) - tl.log(y_val + eps))

    if reduction_mode == _REDUCTION_MODE_NONE:
        # Element-wise: just store the results
        tl.store(out_ptr + offsets * stride_out, out_val, mask=mask)
    else:
        # Reduction modes: compute partial sum in the block, then atomic add
        acc = tl.sum(out_val, axis=0)
        # Atomically add to out_ptr[0]
        tl.atomic_add(out_ptr, acc)


# -----------------------------------------------------------
# 4) KLDivergence Backward Kernel
#    The instructions specify:
#    - For non-log targets: negative of the target.
#    - For log targets: negative of (exp(target) * target).
# -----------------------------------------------------------

@triton.jit
def _kldiv_kernel_backward(
    gt_ptr, grad_ptr,
    n_elements,
    stride_gt, stride_grad,
    log_target, 
    reduction_mode,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load ground truth
    gt_val = tl.load(gt_ptr + offsets * stride_gt, mask=mask, other=0.0)

    # Compute gradient wrt y
    # According to the given instructions, for non-log targets:
    #   grad = - gt
    # For log targets:
    #   grad = - exp(gt) * gt
    # (This is a simplified form as stated in the prompt.)
    if log_target != 0:
        grad_out_val = -tl.exp(gt_val) * gt_val
    else:
        grad_out_val = -gt_val

    if reduction_mode == _REDUCTION_MODE_NONE:
        tl.store(grad_ptr + offsets * stride_grad, grad_out_val, mask=mask)
    else:
        # For sum, mean, batchmean the gradient is computed elementwise anyway,
        # so we store it in grad_ptr. Summation or scaling will happen outside.
        tl.store(grad_ptr + offsets * stride_grad, grad_out_val, mask=mask)


# -----------------------------------------------------------
# 5) Forward wrapper
# -----------------------------------------------------------

def kldiv_forward_triton(y, gt, reduction='mean'):
    """
    y: predictions tensor
    gt: ground truth tensor
    reduction: one of ['none', 'sum', 'mean', 'batchmean']
    """
    reduction_mode = _str_to_reduction_mode[reduction]
    assert y.is_cuda and gt.is_cuda, "Tensors must be on CUDA"
    assert y.shape == gt.shape, "Shapes of y and gt must match"

    n_elements = y.numel()
    dtype = y.dtype

    # Decide block size
    BLOCK_SIZE = min(MAX_FUSED_SIZE, n_elements)
    num_warps = get_num_warps(B
