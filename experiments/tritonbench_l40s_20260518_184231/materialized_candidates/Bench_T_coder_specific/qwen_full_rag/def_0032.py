import torch
import triton
import triton.language as tl
from torch._inductor.ir import Matmul
from torch._inductor.triton_heuristics import pointwise, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


@reduction(
    size_hints=[8192, 512],
    reduction_hint=tl.ReductionHint.DEFAULT,
    filename=__file__,
    meta={
        'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'},
        'device': 0, 'constants': {}, 'mutated_arg_names': [],
        'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 8192
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = triton_helpers.promote_to_tensor(tmp0)
        tl.device_assert((0 <= tmp1) & (tmp1 < 512), "index out of bounds: 0 <= tmp1 < 512")
        tmp2 = tl.load(in_ptr1 + (r1 + (512 * tmp0)), rmask & xmask, eviction_policy='evict_last')
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = _tmp5
    for r in range(1, 16):
        _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = rindex < rnumel
            r1 = rindex
            tmp7 = tl.load(in_ptr2 + (r1 + (512 * tmp0)), rmask & xmask, eviction_policy='evict_last')
            tmp8 = tmp7.to(tl.float32)
            tmp9 = tmp8 * tmp8
            tmp11 = _tmp11 + tmp9
            _tmp11 = tl.where(rmask & xmask, tmp11, _tmp11)
        tmp10 = _tmp11
        tmp5 = tmp5 + tmp10
    tmp5 = tmp5 * 1.953125e-07
    tl.store(out_ptr1 + (x0 + (8192 * tmp0)), tmp5, xmask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf0 = torch.empty((8192, 512), dtype=torch.bfloat16, device='cuda')
    stream0 = torch.cuda.Stream()
    with torch.cuda.stream(stream0):
        triton_(arg0_1, arg1_1, arg2_1, buf0, 8192, 512, XBLOCK=512, RBLOCK=128)
    del arg0_1
    del arg1_1
    del arg2_1
    stream0.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    return (buf0, )


def wrapper(func, inductor_meta):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        assert len(args) == 3
        assert len(kwargs) == 0
        buf0, = args
        args.clear()
        arg0_1 = torch.empty((512, ), dtype=torch.int32, device='cuda')
        torch.arange(0, 512, out=arg0_1)
        arg1_1 = torch.empty((512, 512), dtype=torch.bfloat16, device='cuda')
        fill_diagonal_(arg1_1, 1)
        arg2_1 = torch.empty((512, 512), dtype=torch.bfloat16, device='cuda')
        matmul_triton(arg1_1, arg1_1.T, out=arg2_1)
        del arg1_1
        buf1, = call((arg0_1, arg2_1, buf0))
        del arg0_1
        del arg2_1
        return (buf1, )

    return wrapped_func
