import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    p_ptr,
    grad_ptr,
    momentum_ptr,
    lr,
    weight_decay,
    momentum,
    dampening,
    nesterov,
    n_elements,
    BLOCK_SIZE,  # tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters, gradients, and momentum buffers
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    mom = tl.load(momentum_ptr + offsets, mask=mask)

    # Apply weight decay
    p = p * (1 - lr * weight_decay)

    # Update momentum buffer
    mom = mom * momentum + grad * (1 - dampening)

    # Calculate update direction
    if nesterov:
        update = mom + momentum * grad
    else:
        update = mom

    # Perform the update
    if not maximize:
        p = p - lr * update
    else:
        p = p + lr * update

    # Store the updated parameters and momentum buffers
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(momentum_ptr + offsets, mom, mask=mask)


def sgd_triton(
    params: tl.tensor,
    grads: tl.tensor,
    momentums: tl.tensor,
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    assert all(t.is_cuda for t in (params, grads, momentums))
    n_elements = params.numel()
    BLOCK_SIZE = 512  # Example block size, can be adjusted

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    sgd_kernel[grid](
        params,
        grads,
        momentums,
        lr,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        n_elements,
        BLOCK_SIZE,
    )
