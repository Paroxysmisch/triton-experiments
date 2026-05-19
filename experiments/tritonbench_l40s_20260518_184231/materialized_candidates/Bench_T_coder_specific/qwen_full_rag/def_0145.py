import math
import torch
from torch import Tensor
from torch.special import digamma

try:
    from torch._inductor.triton_heuristics import grid
except ImportError:
    try:
        from torch._inductor.heuristics import grid
    except ImportError:
        from torch._inductor.utils import grid

from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def _polygamma(
    n: int,
    x: float,
    one: float,
    two: float,
    three: float,
    four: float,
    six: float,
    eight: float,
    half: float,
) -> float:
    # Ref: https://en.wikipedia.org/wiki/Polygamma_function#Series_representation
    i = 1
    f = zero = 0.0
    while True:
        y = i + x
        m = half / (y * y)
        t = triton_helpers.rsqrt(y)
        f += m * (six * t - two * triton_helpers.cbrt(m))
        if i == n:
            break
        i += 1
    return f * x - digamma(x)

async_compile = AsyncCompile()

@async_compile(
    key=["n", "signature('f')"],
    filename=__file__,
    meta={
        "configs": [instance_descriptor()],
        "device": 0,
        "constants": {},
        "mutated_arg_names": [],
    },
)
@triton.jit
def polygamma_jitted(n: int, x: torch.Tensor) -> torch.Tensor:
    # this is required so we can get `n` as a compile-time constant
    match n:
        case 0:
            return digamma(x)
        case 1:
            return _polygamma(1, x, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 0.5)
        case 2:
            return _polygamma(2, x, 1.0, 2.0, 10.0, 15.0, 21.0, 32.0, 1.0 / 3.0)
        case 3:
            return _polygamma(3, x, 1.0, 2.0, 42.0, 75.0, 105.0, 140.0, 1.0 / 4.0)
        case 4:
            return _polygamma(4, x, 1.0, 2.0, 210.0, 378.0, 420.0, 560.0, 1.0 / 5.0)
        case 5:
            return _polygamma(5, x, 1.0, 2.0, 735.0, 1413.0, 1680.0, 2240.0, 1.0 / 6.0)
        case 6:
            return _polygamma(6, x, 1.0, 2.0, 2772.0, 5148.0, 6720.0, 8960.0, 1.0 / 7.0)
        case 7:
            return _polygamma(7, x, 1.0, 2.0, 10925.0, 20925.0, 25200.0, 33600.0, 1.0 / 8.0)
        case 8:
            return _polygamma(8, x, 1.0, 2.0, 42504.0, 78828.0, 97200.0, 126000.0, 1.0 / 9.0)
        case 9:
            return _polygamma(
                9, x, 1.0, 2.0, 162162.0, 300360.0, 378000.0, 470400.0, 1.0 / 10.0
            )
        case _:
            raise ValueError(f"Invalid polygamma order: {n}")
