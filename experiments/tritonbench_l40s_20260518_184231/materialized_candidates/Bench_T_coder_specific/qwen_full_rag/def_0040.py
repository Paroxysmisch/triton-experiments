import torch
import triton
import triton.language as tl
from torch import Tensor
from .batch_norm import batch_norm

@triton.jit
def sigmoid_batch_norm_triton(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    if training:
        mean, var = tl.calculate_first_moment(input, eps)
        inv_std = tl.inv(tl.sqrt(var))
        running_mean = tl.update_running_stat(running_mean, mean, momentum)
        running_var = tl.update_running_stat(running_var, var, momentum)
    else:
        inv_std = tl.inv(tl.sqrt(running_var + eps))

    if weight is None:
        weight = tl.full((input.shape[-1], ), 1., dtype=tl.float32)

    if bias is None:
        bias = tl.zeros((input.shape[-1], ), dtype=tl.float32)

    out = tl.batch_norm_sigmoid_kernel(
        input, running_mean, inv_std, weight, bias
    )
    return out

def sigmoid_batch_norm(
    input: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Tensor = None,
    bias: Tensor = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> Tensor:
    args = (input, running_mean, running_var, weight, bias, training, momentum, eps)
    kwargs = {}
    return run("sigmoid_batch_norm", args, kwargs, force_torch=True)
