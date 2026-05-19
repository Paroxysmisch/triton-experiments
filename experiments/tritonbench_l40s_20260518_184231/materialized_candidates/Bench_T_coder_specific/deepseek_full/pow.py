import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import function_heuristics
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@function_heuristics(
    {
        "matches": [
            lambda args: (
                isinstance(args["exponent"], float)
                and args["input"].ndim <= 3
                and (
                    triton_helpers.dyno_enabled()
                    or (
                        args["input"].dtype in [torch.float16, torch.bfloat16]
                        and args["exponent"] == 0.5
                    )
                )
            )
        ],
        "mutations": [],
        "pre_hook": lambda args: PowConstExponentPreHook(args["exponent"]),
    },
)
@triton.jit
def pow(input, exponent, *, out=None):
    r"""Takes the power of each element in input with exponent and returns a
    tensor with the result.

    Args:数组的长度
        input (Tensor): the input tensor.
        exponent (float or tensor): the exponent value.
        Keyword args:
        out (Tensor, optional): the output tensor.

    Examples::

        >>> torch.pow(torch.tensor([1., 2., 3.]), 2)
        tensor([1., 4., 9.])
        >>> torch.pow(torch.tensor([1., 2., 3.]), 2, out=torch.tensor([0.], dtype=torch.float32))
        tensor([1., 4., 9.])
        >>> torch.tensor([1., 2., 3.], dtype=torch.float32, out=torch.tensor([0.], dtype=torch.float32))
        tensor([1., 2., 3.], dtype=torch.float32, device='cuda:0')
        >>> torch.tensor([1., 2., 3.], dtype=torch.float32, out=torch.tensor([0.], dtype=torch.float16))
        tensor([1., 2., 3.], dtype=torch.float16, device='cuda:0')
        >>> torch.pow(torch.tensor([1., 2., 3.], dtype=torch.float32, device='cuda:0'), 2)
        tensor([1., 4., 9.], dtype=torch.float32, device='cuda:0')
    """
    pass

class PowConstExponentPreHook:
    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, args, kwargs):
        args[1] = torch.tensor(self.exponent, device=args[0].device)
        kwargs["exponent"] = torch.tensor(self.exponent, device=args[0].device)
        return args, kwargs
