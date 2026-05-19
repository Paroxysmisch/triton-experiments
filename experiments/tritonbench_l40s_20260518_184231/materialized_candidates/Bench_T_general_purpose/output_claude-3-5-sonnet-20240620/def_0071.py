# triton_sgd.py

import triton
import triton.language as tl

@triton.jit
def sgd_kernel(params_ptr, lr, momentum, weight_decay, dampening, nesterov, maximize, t, num_params):
    # Load parameters and gradients
    params = tl.load(params_ptr + tl.arange(0, num_params) * 4)  # Assuming float32
    g_t = tl.load(params_ptr + tl.arange(0, num_params) * 4 + num_params * 4)  # Gradient

    # Weight decay
    if weight_decay != 0:
        g_t += weight_decay * params

    # Momentum buffer
    if t > 1:
        b_t = momentum * tl.load(params_ptr + tl.arange(0, num_params) * 4 + 2 * num_params * 4) + (1 - dampening) * g_t
    else:
        b_t = g_t

    # Nesterov momentum
    if nesterov:
        g_t += momentum * b_t

    # Update parameters
    if maximize:
        params -= lr * g_t
    else:
        params += lr * g_t

    # Store updated parameters
    tl.store(params_ptr + tl.arange(0, num_params) * 4, params)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    # Prepare parameters for the kernel
    num_params = len(params)
    params_ptr = tl.ptr(params)

    # Launch the Triton kernel
    sgd_kernel[(num_params,)](params_ptr, lr, momentum, weight_decay, dampening, nesterov, maximize, t=1, num_params=num_params)
