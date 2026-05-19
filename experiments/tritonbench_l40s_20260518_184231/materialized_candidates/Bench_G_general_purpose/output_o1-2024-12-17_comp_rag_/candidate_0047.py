import triton
import triton.language as tl
import torch

@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 67108864
    xnumel = 4
    yoffset = tl.program_id(1) * (tl.program_id(2) + 1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128 * x2) + (512 * y1)), mask=xmask & ymask, eviction_policy='evict_last')
    tmp0 = tl.where(tmp0 > 0, tmp0, 0.0)
    tl.store(out_ptr0 + (x2 + (4 * y3)), tmp0, mask=xmask & ymask)

def run_triton(x, y):
    x = torch.randn((67108864, 4), device="cuda")
    y = torch.empty((32768, 4), dtype=torch.float32, device='cuda')
    triton_(x, y, YBLOCK=67108864, XBLOCK=4)
    return y
