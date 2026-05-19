import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    if not isinstance(start, (float, int)):
        raise TypeError("'start' must be a real number or tensor")
    if not isinstance(end, (float, int)):
        raise TypeError("'end' must be a real number or tensor")
    if steps < 1:
        raise ValueError("'steps' must be greater than 0")

    if dtype is None:
        if isinstance(start, (float, int)) and isinstance(end, (float, int)):
            dtype = torch.get_default_dtype()
        else:
            dtype = torch.complex_dtype(torch.get_default_dtype())

    if device is None:
        device = torch.device("cuda") if dtype is torch.float16 or dtype is torch.bfloat16 or dtype is torch.float32 else torch.device("cpu")

    if requires_grad:
        requires_grad = False

    if out is None:
        out = torch.empty((steps,), dtype=dtype, device=device, requires_grad=requires_grad, layout=layout)

    if not start.is_floating_point() and not end.is_floating_point():
        raise RuntimeError("logspace only supports floating point dtype for start and end")

    if not start.is_floating_point():
        start = start.to(dtype)
    if not end.is_floating_point():
        end = end.to(dtype)

    if not triton_helpers.is_power_of_two(steps):
        raise RuntimeError("'steps' must be a power of two for Triton to be efficient")

    triton_logspace[grid(steps)](
        start, end, steps, base, out,
        dtype_real=tl.real(dtype),
        dtype_frac=tl.frac(dtype),
        device=device
    )
    return out


@triton.jit
def triton_logspace(start, end, steps, base, out, dtype_real, dtype_frac, device):
    idx = tl.program_id(0)
    if dtype_frac == dtype_real:
        frac = tl.math.log(base) * (end - start) / (steps - 1)
        exp_start = tl.math.exp(start * tl.math.log(base))
        exp_frac = tl.math.exp(frac * idx)
        value = exp_start * exp_frac
    else:
        frac = tl.math.log(base) * (tl.complex(end, 0) - tl.complex(start, 0)) / (steps - 1)
        exp_start = tl.math.exp(tl.complex(start, 0) * tl.math.log(base))
        exp_frac = tl.math.exp(tl.complex(frac, 0) * idx)
        value = exp_start * exp_frac
    out_idx = idx if dtype_frac == dtype_real else tl.arange(0, 2)
    tl.store(out + out_idx, value, mask=out_idx < steps)


func_inputs = [{'start': 0, 'end': 1, 'steps': 5, 'base': 10.0}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 2.0}, {'start': 0.1, 'end': 1.2, 'steps': 2097152, 'base': 2.0}, {'start': 0.1, 'end': 1.2, 'steps': 2097152, 'base': 10.0}, {'start': 0.1, 'end': 1.2, 'steps': 2097152, 'base': 2.0, 'out': torch.empty(2097152)}, {'start': torch.tensor(0.1), 'end': torch.tensor(1.2), 'steps': 5, 'base': 10.0}, {'start': torch.tensor(0.1, dtype=torch.float64), 'end': torch.tensor(1.2, dtype=torch.float64), 'steps': 5, 'base': 10.0}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0, 'dtype': torch.float32}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0, 'dtype': torch.float64}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0, 'device': torch.device("cpu")}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0, 'device': torch.device("cuda")}, {'start': 0.1, 'end': 1.2, 'steps': 5, 'base': 10.0, 'requires_grad': True}]


for fi in func_inputs:
    start = fi['start']
    end = fi['end']
    steps = fi['steps']
    base = fi['base']
    out = fi['out'] if 'out' in fi else None
    dtype = fi['dtype'] if 'dtype' in fi else None
    device = fi['device'] if 'device' in fi else None
    requires_grad = fi['requires_grad'] if 'requires_grad' in fi else False
    del fi

    ret_triton = logspace(start, end, steps, base, out=out, dtype=dtype, device=device, requires_grad=requires_grad)
    del ret_triton
