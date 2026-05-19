import torch
import triton
import triton.language as tl

# Triton kernel for fused Adam optimizer
@triton.jit
def fused_adam_kernel(
    params_ptr, grads_ptr, n_ele, m_ptr, v_ptr, max_v_ptr, lr, 
    beta1, beta2, beta1_pow_step, beta2_pow_step, 
    eps, wd, step_count, maximize, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    max_v = tl.load(max_v_ptr + offsets, mask=mask) if max_v_ptr else 0.0

    grads += wd * params

    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    m_new_corrected = m_new / (1 - beta1_pow_step)
    v_new_corrected = v_new / (1 - beta2_pow_step)

    if max_v_ptr:
        max_v = tl.maximum(max_v, v_new_corrected)
        v_new_corrected = max_v

    update = lr * m_new_corrected / (tl.sqrt(v_new_corrected) + eps)
    if maximize:
        params_new = params + update
    else:
        params_new = params - update

    tl.store(params_ptr + offsets, params_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)
    if max_v_ptr:
        tl.store(max_v_ptr + offsets, max_v, mask=mask)

# Class to encapsulate the fused Adam optimizer logic
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.step_count = 0

        self.init_moments()

    def init_moments(self):
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        if self.amsgrad:
            self.max_v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        self.step_count += 1
        beta1, beta2 = self.betas
        beta1_pow_step = beta1 ** self.step_count
        beta2_pow_step = beta2 ** self.step_count

        if self.fused:
            self.fused_step(beta1, beta2, beta1_pow_step, beta2_pow_step)
        elif self.foreach:
            self.foreach_step(beta1, beta2, beta1_pow_step, beta2_pow_step)
        else:
            self.for_loop_step(beta1, beta2, beta1_pow_step, beta2_pow_step)

    def fused_step(self, beta1, beta2, beta1_pow_step, beta2_pow_step):
        with torch.no_grad():
            for param, grad, m, v, max_v in zip(self.params, [p.grad for p in self.params], self.m, self.v, self.max_v if self.amsgrad else [None] * len(self.params)):
                grid = lambda meta: (triton.cdiv(param.numel(), meta['BLOCK_SIZE']), )
                fused_adam_kernel[grid](
                    param, grad, param.numel(), m, v, max_v, self.lr, 
                    beta1, beta2, beta1_pow_step, beta2_pow_step, 
                    self.eps, self.weight_decay, self.step_count, self.maximize, 
                    BLOCK_SIZE=1024
                )

    def foreach_step(self, beta1, beta2, beta1_pow_step, beta2_pow_step):
        with torch.no_grad():
            for param, grad, m, v, max_v in zip(self.params, [p.grad for p in self.params], self.m, self.v, self.max_v if self.amsgrad else [None] * len(self.params)):
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1_pow_step)
                v_hat = v / (1 - beta2_pow_step)
                if self.amsgrad:
                    max_v = torch.maximum(max_v, v_hat)
                    v_hat = max_v
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                if self.maximize:
                    param.add_(update)
                else:
                    param.sub_(update)

    def for_loop_step(self, beta1, beta2, beta1_pow_step, beta2_pow_step):
        with torch.no_grad():
            for param, grad, m, v, max_v in zip(self.params, [p.grad for p in self.params], self.m, self.v, self.max_v if self.amsgrad else [None] * len(self.params)):
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1_pow_step)
                v_hat = v / (1 - beta2_pow_step)
                if self.amsgrad:
                    max_v = torch.maximum(max_v, v_hat)
                    v_hat = max_v
                update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                if self.maximize:
                    param.add_(update)
                else:
                    param.sub_(update)

# Example usage:
# params = [torch.randn(10, 10, requires_grad=True, device='cuda')]
# optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True, foreach=True, maximize=False, capturable=False, differentiable=False, fused=True)
# optimizer.zero_grad()
# optimizer.step()
