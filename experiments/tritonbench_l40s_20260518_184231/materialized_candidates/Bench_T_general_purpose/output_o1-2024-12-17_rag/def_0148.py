import torch
import triton
import triton.language as tl

@triton.jit
def _fused_adam_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr, vhat_ptr,
    n_elements,
    lr, beta1, beta2,
    beta1_pow, beta2_pow,
    eps, weight_decay, maximize, amsgrad,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)

    # If maximizing, flip sign of the grad
    if maximize:
        grad = -grad

    # Apply weight decay
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad
    # Update biased second moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad * grad)

    # AMSGrad
    if amsgrad:
        v_hat_old = tl.load(vhat_ptr + offsets, mask=mask)
        vhat_new = tl.maximum(v_hat_old, v_new)
        vhat_corrected = vhat_new / (1 - beta2_pow)
        tl.store(vhat_ptr + offsets, vhat_new, mask=mask)
    else:
        vhat_corrected = v_new / (1 - beta2_pow)

    m_hat = m_new / (1 - beta1_pow)
    denom = tl.sqrt(vhat_corrected) + eps
    param_new = param - lr * (m_hat / denom)

    tl.store(param_ptr + offsets, param_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)


class _AdamFused(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize
        )
        super().__init__(params, defaults)
        self._step = 0
        # Flatten the parameters and associated buffers for a fused update
        self._init_param_storage()

    def _init_param_storage(self):
        # Flatten parameters
        self._params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                self._params.append(p)

        self._flattened_params = torch.cat([p.data.view(-1) for p in self._params])
        self._flattened_grads = torch.cat([p.grad.data.view(-1) for p in self._params])
        self._m = torch.zeros_like(self._flattened_params)
        self._v = torch.zeros_like(self._flattened_params)
        self._vhat = torch.zeros_like(self._flattened_params)

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1

        offset = 0
        # Update grads from actual parameters
        for group in self.param_groups:
            for p in group["params"]:
                n_elems = p.numel()
                self._flattened_grads[offset:offset+n_elems].copy_(p.grad.view(-1))
                offset += n_elems

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            grid = lambda meta: (triton.cdiv(self._flattened_params.numel(), meta['BLOCK_SIZE']),)
            _fused_adam_kernel[grid](
                self._flattened_params, self._flattened_grads,
                self._m, self._v, self._vhat if amsgrad else self._v,
                self._flattened_params.numel(),
                lr, beta1, beta2,
                beta1**self._step, beta2**self._step,
                eps, weight_decay, maximize, amsgrad,
                BLOCK_SIZE=1024
            )

        offset = 0
        # Copy back updated params
        for group in self.param_groups:
            for p in group["params"]:
                n_elems = p.numel()
                p.data.copy_(self._flattened_params[offset:offset+n_elems].view_as(p))
                offset += n_elems

        return None


def Adam(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False,
    foreach=None,
    maximize=False,
    capturable=False,
    differentiable=False,
    fused=None
):
    """
    Implements the Adam optimization algorithm. Optionally supports:
        - AMSGrad variant
        - weight decay
        - maximizing (instead of minimizing)
        - fused implementation for faster CUDA kernels
    """
    # If user specifies fused is True, return the fused version
    if fused:
        return _AdamFused(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize
        )
    else:
        # Otherwise, fall back to standard torch Adam
        # (ignoring foreach, capturable, differentiable for simplicity)
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
