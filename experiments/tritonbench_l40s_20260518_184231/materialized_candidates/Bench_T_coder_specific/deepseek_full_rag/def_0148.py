v_t/(1-\beta_2^t); \theta_t = \theta_{t-1} - \gamma \widehat{m_t}/(\sqrt{\widehat{v_t}} + \epsilon)
other: The foreach and fused implementations are typically faster than the for-loop, single-tensor implementation. The algorithm is based on the paper 'Adam: A Method for Stochastic Optimization'.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
import torch
import triton
import triton.language as tl

@triton.jit
def fused_adam_kernel(
    params_ptr, grads_ptr, n_ele, m_ptr, v_ptr, lr, 
    beta1, beta2, beta1_pow_step, beta2_pow_step, 
    eps, wd, step_count, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    grads += wd * params

    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    m_new_corrected = m_new / (1 - beta1_pow_step)
    v_new_corrected = v_new / (1 - beta2_pow_step)

    params_new = params - (lr * m_new_corrected / (tl.sqrt(v_new_corrected) + eps))

    tl.store(params_ptr + offsets, params_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

class AdamFused:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        self.parameters = list(parameters)
        self.n_ele = sum(param.numel() for param in self.parameters)
        self.params = None
        self.grads = None
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.step_count = 0
        
        self.init_params_and_grads()
        self.init_moments()

    def init_params_and_grads(self):
        self.params = torch.zeros(self.n_ele, dtype=self.parameters[0].dtype, device=self.parameters[0].device)
        self.grads = torch.zeros(self.n_ele, dtype=self.parameters[0].dtype, device=self.parameters[0].device)

        i = 0
        for param in self.parameters:
            num_ele =  param.numel()
            # Populate self.params list
            self.params[i : i+num_ele] = param.view(-1)
            # Ensure that original model will be updated 
            # on updating self.params
            param.data = self.params[i : i+num_ele].view(param.data.shape)
            param.grad = self.grads[i : i+num_ele].view(param.data.shape)

            i += num_ele

        self.params.grad = self.grads

    def init_moments(self):
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)

    def zero_grad(self, set_to_none=False):
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                if self.params.grad.grad_fn is not None:
                    self.params.grad.detach_()
                else:
                    self.params.grad.requires_grad_(False)
                self.params.grad.zero_()

    def step(self):
        self.step_count += 1

        with torch.no_grad():
            grid = lambda meta: (triton.cdiv(self.n_ele, meta['BLOCK_SIZE']), )
            fused_adam_kernel[grid](
                self.params, self.grads, self.n_ele, self.m, self.v, self.lr, 
                self.beta1, self.beta2, self.beta1 ** self.step_count, 
                self.beta2 ** self.step_count, self.eps, self.wd, self.step_count, 
                BLOCK_SIZE=1024
            )
