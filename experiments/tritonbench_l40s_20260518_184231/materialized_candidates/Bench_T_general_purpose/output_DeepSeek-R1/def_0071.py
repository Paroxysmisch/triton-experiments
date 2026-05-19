import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr, grad_ptr, buf_ptr,
    lr, momentum_coef, weight_decay, dampening,
    nesterov, maximize, first_step,
    has_weight_decay: tl.constexpr, has_momentum: tl.constexpr,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    if has_weight_decay:
        grad += weight_decay * param

    if has_momentum:
        buf = tl.load(buf_ptr + offsets, mask=mask)
        if not first_step:
            new_buf = momentum_coef * buf + (1 - dampening) * grad
        else:
            new_buf = grad  # first_step, buffer is initialized to grad
        tl.store(buf_ptr + offsets, new_buf, mask=mask)
        if nesterov:
            update_grad = grad + momentum_coef * new_buf
        else:
            update_grad = new_buf
    else:
        update_grad = grad

    delta = lr * update_grad
    if maximize:
        param += delta
    else:
        param -= delta
    tl.store(param_ptr + offsets, param, mask=mask)

class SGD:
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0, dampening=0,
                 nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.differentiable = differentiable
        self.fused = fused
        self.state = dict()

        if self.nesterov and (self.momentum == 0 or self.dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data
            if not grad.is_contiguous():
                grad = grad.contiguous()
            
            has_momentum = self.momentum != 0
            has_weight_decay = self.weight_decay != 0
            buf = None
            first_step = False

            if has_momentum:
                if id(param) not in self.state:
                    self.state[id(param)] = {'momentum_buffer': torch.clone(grad).detach()}
                    first_step = True
                else:
                    first_step = False
                buf = self.state[id(param)]['momentum_buffer']
                if not buf.is_contiguous():
                    buf = buf.contiguous()

            n_elements = param.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            kernel_args = {
                'param_ptr': param.data_ptr(),
                'grad_ptr': grad.data_ptr(),
                'buf_ptr': buf.data_ptr() if has_momentum else 0,
                'lr': self.lr,
                'momentum_coef': self.momentum,
                'weight_decay': self.weight_decay,
                'dampening': self.dampening,
                'nesterov': self.nesterov,
                'maximize': self.maximize,
                'first_step': first_step,
                'has_weight_decay': has_weight_decay,
                'has_momentum': has_momentum,
                'N': n_elements,
                'BLOCK_SIZE': 1024,
            }
            sgd_kernel[grid](**kernel_args)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False,
        maximize=False, foreach=None, differentiable=False, fused=None):
    return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
               dampening=dampening, nesterov=nesterov, maximize=maximize,
               foreach=foreach, differentiable=differentiable, fused=fused)
