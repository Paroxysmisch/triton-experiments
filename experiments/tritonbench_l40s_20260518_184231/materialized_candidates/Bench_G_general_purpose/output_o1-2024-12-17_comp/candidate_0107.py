import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DX, DY, DOUT, OUT,
    N,
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < N

    x_val = tl.load(X + offsets, mask=mask, other=0.0)
    y_val = tl.load(Y + offsets, mask=mask, other=0.0)
    dout_val = tl.load(DOUT + offsets, mask=mask, other=0.0)

    if not RECOMPUTE_OUTPUT:
        out_val = tl.load(OUT + offsets, mask=mask, other=0.0)
    else:
        # Recompute output = x * sigmoid(y)
        sig_y = 1.0 / (1.0 + tl.exp(-y_val))
        out_val = x_val * sig_y
        tl.store(OUT + offsets, out_val, mask=mask)

    sig_y = 1.0 / (1.0 + tl.exp(-y_val))
    dx_val = sig_y * dout_val
    dy_val = x_val * sig_y * (1.0 - sig_y) * dout_val

    tl.store(DX + offsets, dx_val, mask=mask)
    tl.store(DY + offsets, dy_val, mask=mask)


def _swiglu_bwd(xy: torch.Tensor,
                dx: torch.Tensor,
                dy: torch.Tensor,
                dout: torch.Tensor,
                out: torch.Tensor = None,
                recompute_output: bool = False):
    assert xy.is_cuda, "Input XY must be on CUDA."
    hidden_dim = xy.shape[-1] // 2
    x = xy[..., :hidden_dim]
    y = xy[..., hidden_dim:]

    x_contig = x.contiguous()
    y_contig = y.contiguous()
    dx_contig = dx.contiguous()
    dy_contig = dy.contiguous()
    dout_contig = dout.contiguous()

    x_1d = x_contig.view(-1)
    y_1d = y_contig.view(-1)
    dx_1d = dx_contig.view(-1)
    dy_1d = dy_contig.view(-1)
    dout_1d = dout_contig.view(-1)
    if out is None:
        out_contig = torch.empty_like(x_1d)
    else:
        out_contig = out.contiguous().view(-1)

    n_elems = x_1d.numel()
    BLOCK_N = 1024
    grid = ( (n_elems + BLOCK_N - 1) // BLOCK_N, )

    _swiglu_bwd_kernel[grid](
        x_1d, y_1d,
        dx_1d, dy_1d,
        dout_1d, out_contig,
        n_elems,
        BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute_output
    )

    if out is not None:
        out.copy_(out_contig.view(*out.shape))
