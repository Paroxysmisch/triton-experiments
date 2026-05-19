import triton
import triton.language as tl
import torch

@triton.jit
def _bn_sigmoid_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = off_m < N
    mask_n = off_n < C

    # Create 2D indices for reading/writing
    # "row" in range [0, N], "col" in range [0, C]
    # index in the flattened array is row * C + col
    # BN formula uses mean/var/weight/bias by channel (col)
    # input is shaped [N, C] after we potentially flatten in Python
    # see if in valid range
    # We'll broadcast each row index with each column index
    # to compute final pointer offset

    # Load data
    x = tl.load(x_ptr + (off_m.unsqueeze(1) * C + off_n), mask=(mask_m.unsqueeze(1) & mask_n), other=0.0)

    # Per-channel params
    mean_val = tl.load(mean_ptr + off_n, mask=mask_n, other=0.0)
    var_val  = tl.load(var_ptr  + off_n, mask=mask_n, other=0.0)
    w_val    = tl.load(weight_ptr + off_n, mask=mask_n, other=1.0)  # default=1 if out of range
    b_val    = tl.load(bias_ptr   + off_n, mask=mask_n, other=0.0)  # default=0 if out of range

    # Expand mean, var, weight, bias to match x shape
    mean_val = mean_val.unsqueeze(0)
    var_val  = var_val.unsqueeze(0)
    w_val    = w_val.unsqueeze(0)
    b_val    = b_val.unsqueeze(0)

    # BN + sigmoid
    # out = sigmoid( (x - mean) / sqrt(var + eps) * gamma + beta )
    normed = (x - mean_val) * tl.rsqrt(var_val + eps) * w_val + b_val
    out = 1.0 / (1.0 + tl.exp(-normed))

    # Store
    tl.store(out_ptr + (off_m.unsqueeze(1) * C + off_n), out, mask=(mask_m.unsqueeze(1) & mask_n))

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Detect input shape
    if input.dim() == 2:
        # N, C
        N, C = input.shape
        L = 1
    elif input.dim() == 3:
        # N, C, L
        N, C, L = input.shape
    else:
        raise ValueError("Input must be 2D or 3D")

    # Flatten if 3D: shape => [N*L, C]
    x_flat = input if L == 1 else input.view(N * L, C)
    M = x_flat.shape[0]

    # Convert running_mean, running_var, etc. to device if needed
    device = x_flat.device
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)
    if weight is not None:
        weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Compute batch statistics if training
    if training:
        # mean over dim=0 (channels) => shape [C]
        batch_mean = x_flat.mean(dim=0)
        batch_var  = x_flat.var(dim=0, unbiased=False)

        # Update running stats
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(batch_mean * momentum)
            running_var.mul_(1 - momentum).add_(batch_var * momentum)

        cur_mean = batch_mean
        cur_var  = batch_var
    else:
        # Use running stats
        cur_mean = running_mean
        cur_var  = running_var

    # If weight/bias is None, default them for the kernel
    if weight is None:
        weight = torch.ones_like(cur_mean, device=device)
    if bias is None:
        bias = torch.zeros_like(cur_mean, device=device)

    # Prepare output
    out_flat = torch.empty_like(x_flat)

    # Launch Triton kernel
    BLOCK_M = 64
    BLOCK_N = 64
    grid = ( (M + BLOCK_M - 1) // BLOCK_M, (C + BLOCK_N - 1) // BLOCK_N )

    _bn_sigmoid_kernel[grid](
        x_flat, out_flat,
        cur_mean, cur_var, weight, bias,
        M, C,
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    # Reshape back if 3D
    if L == 1:
        return out_flat
    else:
        return out_flat.view(N, C, L)
