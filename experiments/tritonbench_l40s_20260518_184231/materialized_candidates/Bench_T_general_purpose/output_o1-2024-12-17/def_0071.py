import triton
import triton.language as tl

@triton.jit
def _sgd_kernel(
    param_ptr, grad_ptr, buf_ptr,
    lr, momentum, weight_decay, dampening,
    nesterov, maximize, step,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param_vals = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    grad_vals = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    buf_vals = tl.load(buf_ptr + offsets, mask=mask, other=0.0)

    if weight_decay != 0.0:
        grad_vals += weight_decay * param_vals

    if momentum != 0.0:
        is_first_step = step == 1
        new_buf = tl.where(is_first_step, grad_vals, momentum * buf_vals + (1.0 - dampening) * grad_vals)
        use_grad = tl.where(nesterov, grad_vals + momentum * new_buf, new_buf)
        grad_vals = tl.where(momentum != 0.0, use_grad, grad_vals)
        buf_vals = new_buf

    update = tl.where(maximize, param_vals + lr * grad_vals, param_vals - lr * grad_vals)
    tl.store(param_ptr + offsets, update, mask=mask)
    tl.store(buf_ptr + offsets, buf_vals, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if not hasattr(SGD, '_momentum_buffers'):
        SGD._momentum_buffers = {}
    if not hasattr(SGD, '_steps'):
        SGD._steps = {}

    for param in params:
        if param not in SGD._momentum_buffers:
            SGD._momentum_buffers[param] = tl.zeros_like(param)
            SGD._steps[param] = 0
        SGD._steps[param] += 1

        grid = lambda meta: ( (param.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
        triton.run(
            _sgd_kernel,
            grid=grid,
            args=[
                param, param.grad, SGD._momentum_buffers[param],
                lr, momentum, weight_decay, dampening,
                float(nesterov), float(maximize), SGD._steps[param],
                param.numel()
            ],
            num_warps=4,
            BLOCK_SIZE=1024
        )
