import triton
import triton.language as tl

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

MAX_FUSED_SIZE = int(65536 / 4)
next_power_of_2 = triton.next_power_of_2

def get_num_warps(BLOCK_SIZE):
    if   BLOCK_SIZE >= 32768: return 32
    elif BLOCK_SIZE >= 8192:   return 16
    elif BLOCK_SIZE >= 2048:   return 8
    else:                      return 4

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,       y_stride,
    gt_ptr,      gt_stride,
    loss_ptr,    loss_stride,
    n_elements,  log_target, 
    reduction, BLOCK_SIZE : tl.constexpr
):
    pass  # Kernel code goes here assuming y_ptr, gt_ptr, and loss_ptr are pointers

@triton.jit
def _kldiv_kernel_backward(
    grad_out_ptr,  grad_out_stride,
    y_ptr,         y_stride,
    gt_ptr,        gt_stride,
    grad_y_ptr,    grad_y_stride,
    n_elements,    log_target, 
    BLOCK_SIZE : tl.constexpr
):
    pass  # Kernel code goes here assuming grad_out_ptr, y_ptr etc are pointers

def kldiv_forward_triton(y, gt, log_target=False, reduction='mean'):
    assert(reduction in _str_to_reduction_mode.keys())
    n_elements = y.numel()
    BLOCK_SIZE, num_warps = calculate_settings(n_elements) 

    y, gt = y.flatten().cuda(), gt.flatten().cuda()
    loss = torch.empty((n_elements,), dtype = y.dtype, device = "cuda")

    _kldiv_kernel_forward[(n_elements,)](
        y,  y.stride(0),
        gt, gt.stride(0),
        loss, loss.stride(0),
        n_elements, log_target, 
        _str_to_reduction_mode[reduction],
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    return loss.sum() if reduction != 'none' else loss

def kldiv_backward_triton(grad_output, y, gt, log_target=False):
    n_elements = y.numel()
    BLOCK_SIZE, num_warps = calculate_settings(n_elements) 

    y, gt, grad_output = y.flatten().cuda(), gt.flatten().cuda(), grad_output.flatten().cuda()
    grad_y = torch.empty_like(y)

    _kldiv_kernel_backward[(n_elements,)](
        grad_output, grad_output.stride(0),
        y, y.stride(0),
        gt, gt.stride(0),
        grad_y, grad_y.stride(0),
        n_elements, log_target, 
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps  = num_warps,
    )
    return grad_y
