import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(X, Q, Z, N: tl.constexpr):
    # Convert inputs to float32
    x = X.to(tl.float32)
    q = Q.to(tl.float32)
    # Compute the power of q and add it to x
    p = tl.math.pow(q, -x)
    r = p + tl.arange(0, N)

    # Multiply the sum by p to get the final result
    z = tl.math.mul(p, r)
    # Write the result to the output pointer
    tl.store(Z, z)


def zeta(x: torch.Tensor, q: torch.Tensor, *, out=None) -> torch.Tensor:
    # Ensure x and q have the same device, otherwise move q to x's device
    if x.device != q.device:
        q = q.to(x.device)

    # Get the shape of x and prepare an output tensor
    shape = x.shape
    x = x.ravel()
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.ravel()

    # Define the number of warps and configure the kernel launch
    n_warps = 4 if x.numel() <= 8192 else 8
    kwargs = [x, q, out, 8]
    grid = lambda meta: (x.numel(), )
    zeta_kernel[grid](*kwargs)

    # Reshape the output to match the input shape
    return out.reshape(shape)
