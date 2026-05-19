import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x) * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    Lock,  # pointer to the lock
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _rms_norm_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    FINAL_DW,  # pointer to the weights gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)

class EfficientMemoryRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight,
        eps,
        compress_type,
        jpeg_processor,
        dct_processor,
        quantization_shape=64,
        use_4bit=False,
        prune_ratio=0.75,
        iteration=0,
        static_value=0,
    ):
        x = x.contiguous()
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            mean,
            rstd,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.needs_inputs_grad = x.requires_grad or weight.requires_grad
        ctx.compress_type = compress_type
        ctx.quantization_shape = quantization_shape
        
        kth_val = torch.tensor(0.0, device=x.device)

        if compress_type == "NF4":
            x, quant_state = F.quantize_nf4(x)
            ctx.quant_state = quant_state
        elif compress_type == "PRUNE_ROW":
            if iteration < 10:
                kth_val = torch.kthvalue(
                    x.abs().flatten(), int(x.numel() * prune_ratio)
                ).values
            else:
                kth_val = static_value
            mask = x.abs() > kth_val
            x = x * mask
        elif compress_type != "NONE":
            input_shape = x.shape
            ctx.input_shape = input_shape
            if use_4bit:
                x, quant_state = per_block_quantization_4bit(
                    x, input_shape, quantization_shape
                )
            else:
                x, quant_state = per_block_quantization(
                    x, input_shape, quantization_shape
                )
            ctx.quant_state = quant_state

            if compress_type == "PRUNE":
                kth_val = torch.kthvalue(
                    x.abs().flatten(), int(x.numel() * 0.25)
                ).values
                x = torch.where(x.abs() < kth_val, torch.zeros_like(x), x)
                x = naive_adjustment(x, input_shape, quantization_shape)

            if compress_type == "JPEG":
                x = jpeg_compression(x, input_shape, jpeg_processor, quantization_shape)

            elif compress_type == "DCT":
                x = dct_compression(x, input_shape, dct_processor, quantization_shape)

            elif compress_type == "NAIVE":
                x = naive_adjustment(x, input_shape, quantization_shape)

        ctx.mark_non_differentiable(kth_val)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        y = y.contiguous()
        return y, kth_val

    @staticmethod
    def backward(ctx, dy, grad_kth_val):
        x, w, m, v = ctx.saved_tensors
        quantization_shape = ctx.quantization_shape
        dx, dw = None, None

        if ctx.needs_inputs_grad:
            if ctx.compress_type == "NF4":
                x = F.dequantize_nf4(x, ctx.quant_state)
            elif ctx.compress_type != "NONE" and ctx.compress_type != "PRUNE_ROW":
                quant_state = ctx.quant_state
                input_shape = ctx.input_shape
                x = per_block_dequantization(
                    x, input_shape, quant_state, quantization_shape
                )

            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192:
                GROUP_SIZE_M = 96
            if N <= 4096:
                GROUP_SIZE_M = 128
            if N <= 1024:
                GROUP_SIZE_M = 256
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
            _dw = torch.empty(
                (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device
            )
            dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
            dx = torch.empty_like(dy)
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            _rms_norm_bwd_dx_fused[(M,)](
                dx,
                dy,
                _dw,
                x,
                w,
                m,
                v,
                locks,
                x_arg.stride(0),
                N,
                ctx.eps,
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_warps=ctx.num_warps,
            )
            grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
            _rms_norm_bwd_dwdb[grid](
                _dw,
                dw,
                min(GROUP_SIZE_M, M),
                N,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=128,
            )

        return dx, None, None, None, None, None, None, None, None, None, None, None
