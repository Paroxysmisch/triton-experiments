import torch
import triton
import triton.language as tl

@triton.jit
def logit(x, eps=None):
    if eps is not None:
        x = tl.where(x < eps, eps, x)
        x = tl.where(x > 1 - eps, 1 - eps, x)
    return tl.math.log(x / (1 - x))

def test_logit():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32, device='cuda')
    y = torch.log(x / (1 - x))
    z = logit(x)
    assert torch.allclose(z, y)

    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32, device='cuda')
    y = torch.log(x / (1 - x))
    z = logit(x, 0.1)
    assert torch.allclose(z, y)
