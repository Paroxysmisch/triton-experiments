import triton
import triton.language as tl

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DX, DY, DOUT, OUT,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_dxm, stride_dxn,
    stride_dym, stride_dyn,
    stride_doutm, stride_doutn,
    stride_outm, stride_outn,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask)
    y = tl.load(Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, mask=mask)
    dout = tl.load(DOUT + offs_m[:, None] * stride_doutm + offs_n[None, :] * stride_doutn, mask=mask)

    if RECOMPUTE_OUTPUT:
        out = tl.sigmoid(x) * y
        tl.store(OUT + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, out, mask=mask)
    else:
        out = tl.load(OUT + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, mask=mask)

    sigmoid_x = tl.sigmoid(x)
    dx = dout * y * sigmoid_x * (1 + x * (1 - sigmoid_x))
    dy = dout * sigmoid_x

    tl.store(DX + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn, dx, mask=mask)
    tl.store(DY + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, dy, mask=mask)

def _swiglu_bwd(x, y, dout, out=None, recompute_output=False):
    assert x.is_contiguous(), "Input tensor x must be contiguous"
    assert y.is_contiguous(), "Input tensor y must be contiguous"
    assert dout.is_contiguous(), "Input tensor dout must be contiguous"
    
    batch_dims = x.shape[:-1]
    n = x.shape[-1]
    x = x.reshape(-1, n)
    y = y.reshape(-1, n)
    dout = dout.reshape(-1, n)
    
    m, n = x.shape
    
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    
    if out is None or recompute_output:
        out = torch.empty_like(x)
        recompute = True
    else:
        assert out.is_contiguous(), "Output tensor out must be contiguous"
        out = out.reshape(-1, n)
        recompute = False
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    
    _swiglu_bwd_kernel[grid](
        x, y, dx, dy, dout, out,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        dx.stride(0), dx.stride(1),
        dy.stride(0), dy.stride(1),
        dout.stride(0), dout.stride(1),
        out.stride(0), out.stride(1),
        m, n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        RECOMPUTE_OUTPUT=recompute,
    )
    
    dx = dx.reshape(*batch_dims, n)
    dy = dy.reshape(*batch_dims, n)
    out = out.reshape(*batch_dims, n) if recompute_output else None
    
    return dx, dy, out
