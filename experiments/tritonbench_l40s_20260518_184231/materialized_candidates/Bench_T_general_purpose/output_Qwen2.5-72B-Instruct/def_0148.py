import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    params_ptr,  # Pointer to the parameters
    grads_ptr,   # Pointer to the gradients
    exp_avg_ptr, # Pointer to the first moment (m_t)
    exp_avg_sq_ptr, # Pointer to the second moment (v_t)
    max_exp_avg_sq_ptr, # Pointer to the max second moment (AMSGrad)
    step_ptr,    # Pointer to the current step
    lr,          # Learning rate
    beta1,       # Beta1
    beta2,       # Beta2
    eps,         # Epsilon
    weight_decay, # Weight decay
    amsgrad,     # AMSGrad flag
    maximize,    # Maximize flag
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.load(step_ptr)

    # Load parameters, gradients, and moments
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    max_exp_avg_sq = tl.load(max_exp_avg_sq_ptr + offsets, mask=mask)

    # Apply weight decay
    if weight_decay != 0:
        grads = grads + weight_decay * params

    # Compute first moment
    exp_avg = beta1 * exp_avg + (1 - beta1) * grads

    # Compute second moment
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grads * grads

    # Bias correction
    step = tl.load(step_ptr)
    step_t = step + 1
    bias_correction1 = 1 - beta1 ** step_t
    bias_correction2 = 1 - beta2 ** step_t
    exp_avg_hat = exp_avg / bias_correction1
    exp_avg_sq_hat = exp_avg_sq / bias_correction2

    # AMSGrad
    if amsgrad:
        max_exp_avg_sq = tl.max(max_exp_avg_sq, exp_avg_sq_hat)
        exp_avg_sq_hat = max_exp_avg_sq

    # Update parameters
    params = params - lr * exp_avg_hat / (tl.sqrt(exp_avg_sq_hat) + eps)

    # Store updated values
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)
    if amsgrad:
        tl.store(max_exp_avg_sq_ptr + offsets, max_exp_avg_sq, mask=mask)
    tl.store(step_ptr, step_t, mask=mask)

import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(Adam, self).__init__(params, defaults)
        
        self.foreach = foreach
        self.fused = fused
        self.capturable = capturable
        self.differentiable = differentiable

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
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            if self.foreach:
                adam_kernel[params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, group['lr'], beta1, beta2, group['eps'], group['weight_decay'], group['amsgrad'], group['maximize'], BLOCK_SIZE=1024]
            else:
                for i in range(len(params_with_grad)):
                    adam_kernel[params_with_grad[i], grads[i], exp_avgs[i], exp_avg_sqs[i], max_exp_avg_sqs[i] if group['amsgrad'] else None, state_steps[i], group['lr'], beta1, beta2, group['eps'], group['weight_decay'], group['amsgrad'], group['maximize'], BLOCK_SIZE=1024]

        return loss
