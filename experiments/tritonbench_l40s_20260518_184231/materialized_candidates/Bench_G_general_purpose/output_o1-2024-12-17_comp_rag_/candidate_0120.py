import triton.language as tl
import triton
import torch

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16

@triton.jit
def _forward(
    X, OUT, LUT, sizemax, stride_zx, stride_zout, stride_hout, **meta
):
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from LUT
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # block id and column id
    blockid = tl.load(LUT + offset + rbmn * 4 + 0)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    # pointers to X
    px = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    x = tl.load(px, mask=check, other=-float('inf'))
    x = x.to(tl.float32)
    # computation
    c = tl.max(x, axis=0)
    out = tl.log(tl.sum(tl.exp(x - c), axis=0)) + c
    # pointers to OUT
    pout = OUT + pidz * stride_zout + headid * stride_hout + rowid * BLOCK + rxm
    tl.store(pout, out)

@triton.jit
def _backward(X, OUT, DX, DOUT, LUT, sizemax, stride_zx, stride_zout, stride_hout,
              stride_zdx, stride_zdout, stride_hdout, **meta):
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from look-up table
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # bounds checking on lut
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # initialize pointers to block-sparse input
    blockid = tl.load(LUT + offset + rbmn * 4)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    px = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    pdx = DX + pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    pout = OUT + pidz * stride_zout + headid * stride_hout + rowid * BLOCK + rxm
    pdout = DOUT + pidz * stride_zdout + headid * stride_hdout + rowid * BLOCK + rxm
    # Load
    x = tl.load(px, mask=check, other=-float('inf'))
    out = tl.load(pout)
    dout = tl.load(pdout)
    x = x.to(tl.float32)
    out = out.to(tl.float32)
    dout = dout.to(tl.float32)
    # Computation
    dx = dout * tl.exp(-(out - x))
    tl.store(pdx, dx, mask=check)

class _logsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spdims, block, lut, maxlut, n_head, n_row, bench, time):
        out = torch.zeros((x.shape[0], n_head, n_row), dtype=x.dtype, device=x.device)
        # run kernel
        M = x.shape[0]
        meta = {'BLOCK': block}
        grid = lambda opt: [spdims[0] * spdims[1] * block, M]
        _forward[grid](x, out, lut, maxlut, x.stride(0), out.stride(0), out.stride(1),
                       force_nc_cache=True, **meta)

        ctx.save_for_backward(x, out, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        return out

    @staticmethod
    def backward(ctx, dout):
        x, out, lut = ctx.saved_tensors
        dx = torch.zeros_like(x)
        M = x.shape[0]
        grid = lambda opt: [ctx.spdims[0] * ctx.spdims[1] * ctx.block, M]
        _backward[grid](x, out, dx, dout, lut, ctx.maxlut, x.stride(0), out.stride(0),
                        out.stride(1), dx.stride(0), dout.stride(0), dout.stride(1),
                        force_nc_cache=True, BLOCK=ctx.block)
        return dx, None, None, None, None, None, None, None, None
