import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.triton_heuristics import function_descriptor
from torch._inductor.utils import instance_descriptor

@function_descriptor(
    tags=[],
    description="Solves the least squares problem for an overdetermined system of linear equations using QR decomposition. It computes the least squares solution x that minimizes the Euclidean 2-norm |Ax - b|_2, where A is the coefficient matrix and b is the right-hand side vector or matrix.",
    sync_op_name=None,
    mutated_arg_names=[],
    config={
        "wrapper_mode": "default",
        "wrapper_name": "least_squares_qr",
        "wrapper_conf": [],
        "wrapper_kwargs": ["*", "mode='reduced'", "out=None"],
        "wrapper_defaults": ["*", "mode='reduced'", "out=None"],
        "wrapper_kwarg_names": ["A", "b", "mode", "out"],
        "wrapper_post_hooks": [],
        "wrapper_pre_hooks": [],
        "wrapper_meta": {},
        "wrapper_module": "torch.linalg",
        "wrapper_qualified_name": "torch.linalg.least_squares_qr",
        "wrapper_signature": "(A: Tensor, b: Tensor, *, mode: str = 'reduced', out: Union[NoneType, Tensor] = None) -> Tensor",
        "wrapper_doc": "Solves the least squares problem for an overdetermined system of linear equations using QR decomposition. It computes the least squares solution x that minimizes the Euclidean 2-norm |Ax - b|_2, where A is the coefficient matrix and b is the right-hand side vector or matrix.\n\nArgs:\n    A (Tensor): Coefficient matrix of shape (*, m, n), where * is zero or more batch dimensions.\n    b (Tensor): Right-hand side vector or matrix of shape (*, m) or (*, m, k), where k is the number of right-hand sides.\n    mode (str, optional): Determines the type of QR decomposition to use. One of 'reduced' (default) or 'complete'. See torch.linalg.qr for details.\n    out (Tensor, optional): Output tensor. Ignored if None. Default: None.\n\nReturns:\n    Tensor: The least squares solution.\n\nExample:\n    >>> A = torch.randn(3, 2, device='cuda')  # (*, m, n)\n    >>> b = torch.randn(3, 2, device='cuda')  # (*, m) or (*, m, k)\n    >>> x = torch.linalg.least_squares_qr(A, b)\n",
    },
    wrapper_config=[instance_descriptor(["*", "m", "n"], "*", "*", "*", "*")],
)
@triton.jit
def least_squares_qr(A, b, *, mode='reduced', out=None) -> Tensor:
    return torch.linalg.least_squares_qr(A, b, mode=mode, out=out)
