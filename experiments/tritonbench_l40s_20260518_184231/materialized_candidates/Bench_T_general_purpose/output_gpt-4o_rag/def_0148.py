import torch
import triton
import triton.language as tl

# Triton kernel for fused Adam optimizer
@triton.jit
def fused_adam_kernel(
    params_ptr, grads_ptr, m_ptr, v_ptr, max_v_ptr, n_ele,
    lr, beta1, beta2, beta1_pow, beta2_pow, eps, wd, maximize,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    if wd != 0:
        grads += wd * params

    if maximize:
        grads = -grads

    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    if max_v_ptr is not None:
        max_v = tl.load(max_v_ptr + offsets, mask=mask)
        max_v_new = tl.maximum(max_v, v_new)
        v_new_corrected = max_v_new / (1 - beta2_pow)
        tl.store(max_v_ptr + offsets, max_v_new, mask=mask)
    else:
        v_new_corrected = v_new / (1 - beta2_pow)

    m_new_corrected = m_new / (1 - beta1_pow)
    denom = tl.sqrt(v_new_corrected) + eps
    params_new = params - lr * m_new_corrected / denom

    tl.store(params_ptr + offsets, params_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

class AdamOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=False):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.step_count = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.max_v = [torch.zeros_like(p) if amsgrad else None for p in self.params]

    def step(self):
        self.step_count += 1
        beta1_pow = self.beta1 ** self.step_count
        beta2_pow = self.beta2 ** self.step_count

        for p, m, v, max_v in zip(self.params, self.m, self.v, self.max_v):
            if p.grad is None:
                continue

            n_ele = p.numel()
            grid = lambda meta: (triton.cdiv(n_ele, meta['BLOCK_SIZE']),)

            fused_adam_kernel[grid](
                p, p.grad, m, v, max_v, n_ele,
                self.lr, self.beta1, self.beta2, beta1_pow, beta2_pow,
                self.eps, self.weight_decay, self.maximize,
                BLOCK_SIZE=1024
            )

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Example usage
params = [torch.randn(10, device='cuda', requires_grad=True)]
optimizer = AdamOptimizer(params, lr=0.001, amsgrad=True, maximize=False)

# In your training loop
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
