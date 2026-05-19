import functools
import torch
import triton
import triton.language as tl

@triton.jit
def adam_fused(
    p, grad, m, v, h, lr, beta1, beta2, eps, wd, bias_correction, max_mode
):
    # Compute bias-corrected first and second moment estimates
    m = m * beta1 + grad
    v = v * beta2 + grad * grad
    if bias_correction:
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
    else:
        m_hat = m
        v_hat = v
    # Update the parameters
    if wd != 0:
        p = p * (1 - lr * wd)
    if max_mode:
        p = p - lr * m_hat / (tl.sqrt(v_hat) + eps)
    else:
        p = p + lr * m_hat / (tl.sqrt(v_hat) + eps)
    # Store the updated moments and state
    h[0] = m
    h[1] = v
    h[2] = p

def _adam(params, grads, moment1, moment2, state, lr, beta1, beta2, eps, wd, amsgrad,
           bias_correction, foreach, max_mode, out_params):
    # Get the number of parameters and their strides
    n_tensors = moment1.shape[0]
    tensor_strides = moment1.stride(0)
    # Define a function to perform the Adam update for a single parameter
    def _single_tensor_adam(param, grad, m, v, state, lr, beta1, beta2, eps, wd, amsgrad,
                            bias_correction, max_mode):
        # Load the state for the parameter
        m_ptr = state + 3 * param * tensor_strides
        v_ptr = state + 3 * param * tensor_strides + 1
        h_ptr = state + 3 * param * tensor_strides + 2
        # Load the moments and state
        m = tl.load(m_ptr).to(tl.float32)
        v = tl.load(v_ptr).to(tl.float32)
        if amsgrad:
            h = tl.load(h_ptr).to(tl.float32)
        # Compute the Adam update
        adam_fused(
            param, grad, m, v, (m_ptr, v_ptr, h_ptr), lr, beta1, beta2, eps, wd, bias_correction,
            max_mode
        )
    # Define a function to perform the Adam update for a batch of parameters
    def _foreach_adam(param, grad, m, v, state, lr, beta1, beta2, eps, wd, amsgrad, bias_correction,
                      max_mode):
        # Compute the Adam update using the foreach implementation
        adam_fused(
            param, grad, m, v, state, lr, beta1, beta2, eps, wd, bias_correction, max_mode
        )
    # Determine whether to use the foreach implementation
    if foreach and n_tensors > 1024:
        _foreach_adam = functools.partial(_foreach_adam, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                                          wd=wd, amsgrad=amsgrad, bias_correction=bias_correction,
                                          max_mode=max_mode)
        triton.foreach(_foreach_adam, params, grads, moment1, moment2, state)
    else:
        # Apply the Adam update to each parameter
        for i, (param, grad, m, v, h) in enumerate(zip(params, grads, moment1, moment2, state)):
            _single_tensor_adam(
                param, grad, m, v, h, lr, beta1, beta2, eps, wd, amsgrad, bias_correction, max_mode
            )
    # Return the updated parameters if requested
    if out_params is not None:
        out_params.copy_(params)
