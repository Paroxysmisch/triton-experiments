import torch
import triton
import triton.language as tl

from .utils import calculate_settings, get_block_size

def has_fp64_support(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)

def has_bf16_support(dtype):
    return dtype in (torch.bfloat16,)

def cfggen():
    block_m = [1, 2, 4]
    block_n = [1024, 2048, 4096]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs

@triton.jit
def sgd_kernel(
    params_ptr,
    state_ptr,
    grad_ptr,
    new_params_ptr,
    curr_step_ptr,
    lr,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    maximize,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    state_ptr += pid * 2
    params_ptr += pid * BLOCK_SIZE_N
    new_params_ptr += pid * BLOCK_SIZE_N
    grad_ptr += pid * BLOCK_SIZE_N

    params = tl.load(params_ptr, mask=pid < BLOCK_SIZE_M, other=0.0).to(tl.float32)
    state = tl.load(state_ptr, mask=pid < BLOCK_SIZE_M, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr, mask=pid < BLOCK_SIZE_M, other=0.0).to(tl.float32)

    if weight_decay != 0:
        grad += weight_decay * params
    if momentum != 0:
        state = state * momentum + grad
        if nesterov:
            grad += momentum * state
    else:
        state = grad
    if dampening != 0:
        state *= 1 - dampening
    if not maximize:
        params -= lr * state
    else:
        params += lr * state
    tl.store(new_params_ptr, params, mask=pid < BLOCK_SIZE_M)
    tl.store(state_ptr, state, mask=pid < BLOCK_SIZE_M)

class _FusedSGD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        params,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        maximize,
        foreach,
        differentiable,
        fused,
    ):
        if fused is None:
            fused = has_fp64_support(params.dtype) and torch.is_grad_enabled()

        if fused:
            orig_params = params
            if orig_params.requires_grad and not differentiable:
                orig_params = orig_params.detach()
            new_params = torch.empty_like(orig_params)
            curr_step = torch.zeros(1, dtype=torch.int32, device=orig_params.device)
            N, block_size = calculate_settings(
                numels=orig_params.numel(), block_size=get_block_size(orig_params)
            )
            sgd_kernel[N,](
                orig_params,
                torch.zeros(N, 2, dtype=torch.float32, device=orig_params.device),
                torch.zeros(N, block_size, dtype=torch.float32, device=orig_params.device),
                new_params,
                curr_step,
                lr,
                momentum,
                dampening,
                weight_decay,
                nesterov,
                maximize,
            )
            return new_params
        else:
            clone_params = params.detach().clone()
            orig_params = params
            if orig_params.requires_grad and not differentiable:
                orig_params = orig_params.detach()
            grad = torch.zeros_like(orig_params)
            for param, new_param in zip(orig_params, clone_params):
                grad[0] = 0.0
                def closure():
                    nonlocal grad
                    grad += param.grad
                    param.grad = None
                    return param
                SGD(
                    params=[param],
                    lr=lr,
                    momentum=momentum,
                    dampening=dampening,
                    weight_decay=weight_decay,
                    nesterov=nesterov,
                    maximize=maximize,
                    foreach=fused,
                    differentiable=differentiable,
                )(closure())
                new_param.copy_(grad[0])
            return clone_params

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    return _FusedSGD.apply(params, lr, momentum, dampening, weight_decay, nesterov, maximize, foreach, differentiable, fused)
