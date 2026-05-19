import torch
import triton
import triton.language as tl
from torch.optim import Optimizer

@triton.jit
def adam_kernel(
    params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, max_exp_avg_sq_ptr,
    n_elements, lr, beta1, beta2, eps, step, weight_decay,
    maximize, amsgrad,
    BLOCK_SIZE: tl.constexpr
):
    # Grid-stride loop
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

    # Adjust gradient for maximization
    if maximize:
        grads = -grads

    # Weight decay
    if weight_decay != 0:
        grads = grads + weight_decay * params

    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1 - beta1) * grads
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grads * grads)

    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1
    
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask)
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)
        denom = tl.sqrt(max_exp_avg_sq / bias_correction2) + eps
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
    else:
        denom = tl.sqrt(exp_avg_sq / bias_correction2) + eps

    # Update parameters
    params = params - step_size * (exp_avg / denom)

    # Store results
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)

class Adam(Optimizer):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, foreach=foreach, maximize=maximize,
                       capturable=capturable, differentiable=differentiable,
                       fused=fused)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
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

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            beta1, beta2 = group['betas']

            # Determine block size based on tensor size
            BLOCK_SIZE = min(1024, max(128, triton.next_power_of_2(params_with_grad[0].numel())))
            
            for i in range(len(params_with_grad)):
                grid = lambda meta: (triton.cdiv(params_with_grad[i].numel(), meta['BLOCK_SIZE']),)
                
                adam_kernel[grid](
                    params_with_grad[i].data.ptr,
                    grads[i].data.ptr,
                    exp_avgs[i].data.ptr,
                    exp_avg_sqs[i].data.ptr,
                    max_exp_avg_sqs[i].data.ptr if group['amsgrad'] else exp_avg_sqs[i].data.ptr,
                    params_with_grad[i].numel(),
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    state_steps[i],
                    group['weight_decay'],
                    group['maximize'],
                    group['amsgrad'],
                    BLOCK_SIZE=BLOCK_SIZE
                )

        return loss
