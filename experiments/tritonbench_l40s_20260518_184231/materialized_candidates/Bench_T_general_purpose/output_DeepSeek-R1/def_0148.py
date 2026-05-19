import torch
import triton
import triton.language as tl
from torch.optim import Optimizer

@triton.jit
def adam_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr, max_v_hat_ptr,
    beta1, beta2, beta1_pow_t, beta2_pow_t,
    lr, eps, amsgrad,
    param_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < param_size

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    if amsgrad:
        max_v_hat = tl.load(max_v_hat_ptr + offsets, mask=mask, other=0.0)

    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad * grad

    m_hat = new_m / (1 - beta1_pow_t)
    v_hat = new_v / (1 - beta2_pow_t)

    if amsgrad:
        max_v_hat = tl.maximum(max_v_hat, v_hat)
        denom = tl.sqrt(max_v_hat) + eps
    else:
        denom = tl.sqrt(v_hat) + eps

    param_update = lr * m_hat / denom
    new_param = param - param_update

    tl.store(param_ptr + offsets, new_param, mask=mask)
    tl.store(m_ptr + offsets, new_m, mask=mask)
    tl.store(v_ptr + offsets, new_v, mask=mask)
    if amsgrad:
        tl.store(max_v_hat_ptr + offsets, max_v_hat, mask=mask)

class TritonAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, foreach=None, maximize=False,
                 capturable=False, differentiable=False, fused=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize,
                        capturable=capturable,
                        differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('fused', None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grad = p.grad
                if group['maximize']:
                    grad = -grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                grads.append(grad)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0.0, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                else:
                    max_exp_avg_sqs.append(None)
                state['step'] += 1
                state_steps.append(state['step'])

            for param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, step_t in zip(
                params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
            ):
                if grad.is_sparse:
                    raise RuntimeError('TritonAdam does not support sparse gradients')
                step = step_t.item()
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                num_elements = param.numel()
                if num_elements == 0:
                    continue
                grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
                amsgrad = group['amsgrad']
                max_exp_avg_sq_ptr = max_exp_avg_sq.data_ptr() if amsgrad else 0

                adam_kernel[grid](
                    param.data_ptr(),
                    grad.data_ptr(),
                    exp_avg.data_ptr(),
                    exp_avg_sq.data_ptr(),
                    max_exp_avg_sq_ptr,
                    beta1,
                    beta2,
                    bias_correction1,
                    bias_correction2,
                    group['lr'],
                    group['eps'],
                    amsgrad,
                    num_elements,
                    BLOCK_SIZE=1024,
                )

        return loss

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None) -> Optimizer:
    return TritonAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)
