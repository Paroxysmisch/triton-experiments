import torch
import triton
import triton.language as tl

@triton.jit
def fused_adam_kernel(
    params_ptr, grads_ptr, n_ele, m_ptr, v_ptr, v_max_ptr,
    lr, beta1, beta2, beta1_pow_step, beta2_pow_step,
    eps, wd, step_count, amsgrad_flag, maximize_flag,
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

    grads = grads * (1 - 2 * maximize_flag)
    grads += wd * params

    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    m_new_corrected = m_new / (1 - beta1_pow_step)
    v_new_corrected = v_new / (1 - beta2_pow_step)

    v_max = tl.load(v_max_ptr + offsets, mask=mask)
    v_max_new = tl.maximum(v_max, v_new_corrected)
    denominator_ams = tl.sqrt(v_max_new) + eps
    denominator_non_ams = tl.sqrt(v_new_corrected) + eps
    denominator = tl.where(amsgrad_flag, denominator_ams, denominator_non_ams)

    params_new = params - (lr * m_new_corrected / denominator)

    tl.store(params_ptr + offsets, params_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)
    tl.store(v_max_ptr + offsets, tl.where(amsgrad_flag, v_max_new, v_max), mask=mask)

class AdamFused:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=False, capturable=False, differentiable=False):
        self.parameters = list(parameters)
        self.n_ele = sum(p.numel() for p in self.parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.step_count = 0

        if self.differentiable:
            raise NotImplementedError("Differentiable Adam is not supported")

        self._init_params_and_grads()
        self._init_moments()

    def _init_params_and_grads(self):
        device = self.parameters[0].device
        dtype = self.parameters[0].dtype
        self.params = torch.zeros(self.n_ele, dtype=dtype, device=device)
        self.grads = torch.zeros(self.n_ele, dtype=dtype, device=device)

        offset = 0
        for p in self.parameters:
            numel = p.numel()
            self.params[offset:offset+numel] = p.view(-1).detach()
            p.data = self.params[offset:offset+numel].view(p.shape)
            p.grad = self.grads[offset:offset+numel].view(p.shape)
            offset += numel

        self.params.grad = self.grads

    def _init_moments(self):
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)
        self.v_max = torch.zeros_like(self.params) if self.amsgrad else torch.empty(0, device=self.params.device)

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            self.grads = None
            for p in self.parameters:
                p.grad = None
        else:
            if self.grads is not None:
                self.grads.zero_()

    def step(self):
        self.step_count += 1
        if self.differentiable:
            raise RuntimeError("Differentiable Adam is not supported")

        beta1_pow = self.beta1 ** self.step_count
        beta2_pow = self.beta2 ** self.step_count

        amsgrad_flag = 1.0 if self.amsgrad else 0.0
        maximize_flag = 1.0 if self.maximize else 0.0
        v_max_ptr = self.v_max if self.amsgrad else torch.empty(0, device=self.params.device)

        with torch.no_grad():
            grid = lambda meta: (triton.cdiv(self.n_ele, meta['BLOCK_SIZE']),)
            fused_adam_kernel[grid](
                self.params, self.grads, self.n_ele,
                self.m, self.v, v_max_ptr,
                self.lr, self.beta1, self.beta2,
                beta1_pow, beta2_pow,
                self.eps, self.wd, self.step_count,
                amsgrad_flag, maximize_flag,
                BLOCK_SIZE=1024
            )

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
    if fused or (fused is None and foreach is None):
        return AdamFused(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, capturable=capturable, differentiable=differentiable)
    else:
        raise NotImplementedError("foreach and single-tensor implementations are not provided")
