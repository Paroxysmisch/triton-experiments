pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    pout = OUT + pidz * stride_zout + headid * stride_hout + rowid * BLOCK + rxm
    pdout = DOUT + pidz * stride_zdout + headid * stride_hdout + rowid * BLOCK + rxm
    # load data into SRAM
    x = tl.load(px, mask=check, other=-float('inf'))
    out = tl.load(pout)
    dout = tl.load(pdout)
    x = x.to(tl.float32)
    out = out.to(tl.float32)
    dout = dout.to(tl.float32)
    # computation
    # (1) update gradients of input
    # dx = dout * exp(x - out)
    c = tl.exp(x - out)
    dx = c * dout
    tl.store(pdx, dx, mask=check)
    # (2) update output gradients
    # dout = 0
    # This is zeroed out at the start of the backward pass

class BlockSparseLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sparsity_layout, block_size, sparsity_efficiency=0.95):
        ctx.block_size = block_size
        # create look-up table
        luts, num_blocks, num_heads = create_lut(x, sparsity_layout, block_size,
                                                 sparsity_efficiency)
        # run kernel
        B, H, Z, D = x.shape
        out = torch.empty(B, H, Z, D, device=x.device, dtype=x.dtype)
        meta = {'BLOCK': block_size, 'TN': next_power_of_2(D)}
        num_warps = num_warps(D)
        def grid(meta): return (num_blocks * block_size, num_blocks)
        _forward[grid](x, out, luts, num_blocks, Z, meta['BLOCK'], stride_zx=D)
        ctx.save_for_backward(x, out, luts, num_blocks)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, out, luts, num_blocks = ctx.saved_tensors
        B, H, Z, D = x.shape
        # run kernel
        dx = torch.empty_like(x)
        meta = {'BLOCK': ctx.block_size, 'TN': next_power_of_2(D)}
        num_warps = num_warps(D)
        def grid(meta): return (num_blocks * ctx.block_size, num_blocks)
        _backward[grid](x, out, dx, dout, luts, num_blocks, Z, meta['BLOCK'],
                        stride_hout=D, stride_hdout=D)
        return dx, None, None, None

def block_sparse_logsumexp(x, sparsity_layout, block_size, sparsity_efficiency=0.95):
    """
    Arguments:
        x: tensor of shape (batch, heads, n_blocks, block_size)
        sparsity_layout: tensor of shape (n_heads, n_blocks) with integers in the range [0, 3]
        block_size: integer, size of the blocks
        sparsity_efficiency: float, target efficiency for sparsity (default: 0.95)
    Returns:
        out: tensor of shape (batch, heads, n_blocks, block_size)
    """
    return BlockSparseLogSumExp.apply(x, sparsity_layout, block_size, sparsity_efficiency)

import torch
from torch import Tensor
from torch.autograd import Function
from typing import Optional

class _LogSumExp(Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False,
                scale: Optional[float] = None) -> Tensor:
        if dim is None:
            x = x.flatten()
            dim = 0
        else:
            shape = list(x.shape)
            x = x.contiguous()
        out = torch.empty_like(x)
        N = 1
        for i in range(dim):
            N *= shape[i]
            shape[i] = 1
        D = x.numel() // N // shape[dim]
        grid = (N,)
        logsumexp_fwd_kernel[grid](x, out, N, D, dim, scale=scale)
        if not keepdim:
            out = out.reshape(shape)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, dout: Tensor) -> Tensor:
        out, = ctx.saved_tensors
        return logsumexp_bwd_kernel(out, dout), None, None, None

logsumexp = _LogSumExp.apply
