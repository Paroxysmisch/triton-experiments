import torch
import triton
import triton.language as tl
from functools import reduce
import operator

dim2shape = lambda dim, s: reduce(operator.mul, [s[i] for i in dim], 1)
dim2stride = lambda dim, s: [reduce(operator.mul, [s[i] for i in dim[i + 1:]], 1) for i in range(len(dim))]
str2indice = lambda s, d: reduce(operator.add, [s[i + 1] * d[i] for i in range(len(d))])

_ = torch.empty(0)

def validate_dims_and_get_shape(input: torch.Tensor, dim: list = None, ndim: int = None):
    if isinstance(dim, int):
        assert dim >= -input.ndim and dim < input.ndim, f"Invalid dim {dim}"
        if dim < 0:
            dim = dim + input.ndim
    else:
        assert isinstance(dim, list) and len(dim) > 0 and max(dim) < input.ndim and min(dim) > -input.ndim, "Invalid dims"
        if max(dim) < 0:
            dim = [d + input.ndim for d in dim]
        assert len(dim) <= 2, f"Invalid dims length {len(dim)}"
    if ndim is not None:
        assert ndim >= len(dim), f"Invalid ndims {ndim}"

def dim2rshape(dim: list, ndim: int):
    rndim = ndim - len(dim)
    if isinstance(dim, int):
        return [1] * dim + [dim2shape(dim + 1, ndim)]
    else:
        rdim = [1] * rndim
        return rdim + dim2shape(dim, ndim)

def dim1shape(d: int, s: list):
    return [1] * d + s

def dim2rstride(dim: list, s: list):
    rndim = len(dim)
    rs = [0] * rndim
    for i in range(rndim - 1, -1, -1):
        rs[i] = dim2shape(dim[:i + 1], s) + rs[i + 1]
    return rs

def dim2xshape(input: torch.Tensor, dim: list = None, ndim: int = None, keepdim: bool = False):
    sh = torch.tensor(validate_dims_and_get_shape(input, dim, ndim), dtype=torch.int32, device=input.device)
    xsh = dim2shape(dim, sh.size(0)).item()
    return (sh.flip([0]), [xsh], [-1] * xsh + [1] if keepdim else [1] * xsh)

def dim2xstride(input: torch.Tensor, dim: list = None, ndim: int = None):
    sh = torch.tensor(validate_dims_and_get_shape(input, dim, ndim), dtype=torch.int32, device=input.device)
    return dim2xshape(input, dim, sh.size(0))

def dim2rshape_stride(dim: list, s: list):
    rsh = dim2rshape(dim, len(s) + 1)
    rs = dim2rstride(dim, s)
    return rsh, rs

@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, tl.num_programs(0), row_step):
        x_ptr = x_ptr + row_idx * x_row_stride
        y_ptr = y_ptr + row_idx * x_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        x_ptrs = x_ptr + col_offsets
        y_ptrs = y_ptr + col_offsets
        x = tl.load(x_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
        mask = tl.load(mask_ptr + row_idx) if mask_ptr is not None else None
        x = x - tl.max(x, mask=mask, axis=0)
        num = tl.exp(x)
        denom = tl.sum(num, mask=mask, axis=0)
        y = num / denom
        tl.store(y_ptrs, y, mask=col_offsets < n_cols)
        # tl.debug_barrier()

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    validate_dims_and_get_shape(input, dim)
    (sh, xsh, rsh), (s, xs, rs) = dim2xshape(input, dim), dim2xstride(input, dim)
    assert (dim2shape(dim, sh.size(0)) == 1) == (input.ndim > 1), "dim must be a 1D tensor"
    assert input.ndim - 1 <= len(dim), "Invalid dim length"
    assert torch.all(input >= 0), f"logits must be >= 0, but min is {torch.min(input)}"
    if mask is not None:
        validate_dims_and_get_shape(mask, dim)
        assert torch.all(mask == mask.to(torch.bool)), "mask must be a boolean tensor"
    dtype, device = input.dtype, input.device
    numel = input.numel()
    blk = dim2shape(max(dim2shape(dim, sh.size(0)), 64), sh.size(0))
    n_rows = dim2shape(dim, input.size(dim))
    assert blk % 32 == 0, "blk must be a multiple of 32"
    occupancy_grid = lambda g, n: ((n + g - 1) // g, g)
    lcm_denom = lambda a, b: a * b // math.gcd(a, b)
    kwargs = lambda N, *, w, s, blk: [N, w, N // w * s, blk]
    num_warps, n_warps = 1, 1
    occupancy_kwarg = lambda nw, *, g, w, s, blk: kwargs(nw * NUM_REGS // (g * w * lcm_denom(w, blk)), g=g, w=w, s=s, blk=blk)
    kernel = triton.compile(
        mod,
        signature=f"*{dtype},*{dtype},i32,i32,i32,i32",
        device=device,
        constants={"BLOCK_SIZE": blk, "mask_ptr": mask.data_ptr() if mask is not None else None},
        **occupancy_kwarg(
            *occupancy_grid(
                min(
                    max(
                        range(4, 257, 2),
                        key=lambda nw: NUM_REGS // (nw * lcm_denom(nw, blk)) % occupancy_grid(nw, 1)[0] == 0
                    ),
                    key=lambda nw: lcm_denom(nw, blk) // nw % 2 == 1
                ),
                blk
            ),
            g=n_warps, s=1, blk=blk
        )
    )
    kernel[(occupancy_kwarg(num_warps, g=num_warps, w=num_warps, s=1, blk=blk))](
        input, torch.empty_like(input),
        input.stride(dim), input.size(dim), input.size(-1),
        mask.data_
