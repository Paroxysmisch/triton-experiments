t/(1-\beta_2^t); \theta_t = \theta_{t-1} - \gamma \widehat{m_t}/(\sqrt{\widehat{v_t}} + \epsilon)
other: The foreach and fused implementations are typically faster than the for-loop, single-tensor implementation. The algorithm is based on the paper 'Adam: A Method for Stochastic Optimization'.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement a fused Adam optimizer kernel for GPU execution. The kernel uses block-level parallelism to update parameters, gradients, and first and second moments for optimization based on learning rate, beta values, weight decay, and other inputs. Implement an AdamFused class to manage parameters and steps. import torch
import triton
import triton.language as tl

# Triton kernel for fused Adam optimizer
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

# Class to encapsulate the fused Adam optimizer logic
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
</s>
<|user|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Implements the Adam optimization algorithm, which is an adaptive learning rate optimization algorithm designed for training deep neural networks. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. The algorithm can optionally use the AMSGrad variant, apply weight decay, and maximize the objective function. It supports various implementation optimizations like foreach and fused implementations for performance improvements on CUDA.
Wrapper Entry Information: def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None) -> Optimizer
Math: m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t; v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2_t; \widehat{m_t} = m_t/(1-\beta_1^t); \widehat{v_t} = v_t/(1-\beta_2^t); \theta_t = \theta_{t-1} - \gamma \widehat{m_t}/(\sqrt{\widehat{v_t}} + \epsilon)
other: The foreach and fused implementations are typically faster than the for-loop, single-tensor implementation. The algorithm is based on the paper 'Adam: A Method for Stochastic Optimization'.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate. <|system|>Document 1:
Use triton language to implement a fused Adam optimizer kernel for GPU execution. The kernel uses block-level parallelism to update parameters, gradients, and first and second moments for optimization based on learning rate, beta values, weight decay, and other inputs. Implement an AdamFused class to manage parameters and steps. import torch
import triton
import triton.language as tl

# Triton kernel for fused Adam optimizer
@triton.jit
def fused_adam_kernel(
    params_ptr, grads_ptr, n_ele, m_ptr, v_ptr, lr, 
    beta1, beta2, beta1_pow_step, beta2_pow_step, 
    eps, wd, step_count, BLOCK_
