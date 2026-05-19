import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_batch_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, stride, N, C, BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < C, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / C
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, C, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < C, other=0.).to(tl.float32)
        x = tl.where(cols < C, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / C
    rstd = 1 / tl.sqrt(var + 1e-5)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, C, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < C
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Trigonometric transformation
        y = 1 / (1 + tl.exp(-y)))
        # Write output
        tl.store(Y + cols, y, mask=mask)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5) -> torch.Tensor:
    if input.dtype == torch.float16:
        input = input.to(torch.float32)

    C = input.shape[-1]
    N = input.numel() // C

    # Create output
    output = torch.empty_like(input)

    # reshape input data into 2D tensor
    input_2d = input.view(N, C)
    output_2d = output.view(N, C)

    # construct mean and rstd
    mean = torch.empty((N, ), dtype=torch.float32, device='cuda')
    rstd = torch.empty((N, ), dtype=torch.float32, device='cuda')

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))

    if C > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _sigmoid_batch_norm_fwd_fused[(N,)](
        input_2d, output_2d, weight, bias,
        mean, rstd,
        input_2d.stride(0), N, C,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # reshape output
    output = output.view(input.shape)

    if not training:
        std = torch.sqrt(running_var + eps)
        output = (output - running_mean) / std
        output = torch.sigmoid(output * weight + bias)

    return output
