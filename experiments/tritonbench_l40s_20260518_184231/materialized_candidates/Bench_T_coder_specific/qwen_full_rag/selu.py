import triton
import triton.language as tl
import torch
from triton.language.libdevice import exp2f


@triton.jit
def selu_kernel(
    X, 
    Y,
    N,
    BLOCK_SIZE: tl.constexpr
    ):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    
    zero = 0.0
    one = 1.0
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    y = scale * tl.where(x >= zero, x, alpha * (exp2f(x) - one))
    tl.store(Y + offset, y, mask=mask)


class Selu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, inplace=False):
        if not isinstance(A, triton._C.libtorch.TensorView):
            A = triton._C.libtorch.make_tensor(A)
        
        if inplace:
            raise RuntimeError("inplace selu is not supported.")
        
        out = triton._C.empty_like(A)
        N = A.numel()
        
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
        selu_kernel[grid](A, out, N, BLOCK_SIZE=1024)
        
        return out


def selu(A, inplace=False):
    """
    Alias for :func:`selu.Selu.apply`
    """
    return Selu.apply(A, inplace)
