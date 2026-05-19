(acc)

    tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)

@triton.jit
def kernel_bwd(
    dY,  # Gradient of loss w.r.t. output
    dX,  # Gradient of loss w.r.t. input
    W,  # Weights
    act_in,  # Input to activation function (A x W + B)
    dW,  # Gradient of loss w.r.t. weight
    dB,  # Gradient of loss w.r.t. bias
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = grid_n
    pid_m = pid // width
    pid_n = pid % width

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    dY = dY + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    act_in = act_in + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    dW = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    dB = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(act_in)
            b = tl.load(W)
            dy = tl.load(dY)
        else:
            a = tl.load(act_in, mask=rn[None, :] < N, other=0.0)
            b = tl.load(W, mask=rn[:, None] < N, other=0.0)
            dy = tl.load(dY, mask=rn[None, :] < N, other=0.0)
        db = tl.sum(dy, axis=0)
        dw = tl.dot(a[:, None], dy[None, :])
        dW += dw
        dB += db

        if EVEN_K:
            act_in += BLOCK_K
            dY += BLOCK_K
        else:
            act_in += BLOCK_K * stride_ak
            dY += BLOCK_K * stride_ak

    tl.store(dW + rn[None, :], dW)
    tl.store(dB + rn, dB)

def triton_linear_act(input, weight, bias=None, activation="tanh"):
    # Prepare data
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Initialize output tensor
    output = torch.empty_like(input)

    # Compute forward pass
    kernel_fwd[32, 32](
        output,
        input,
        weight,
        bias,
        input.shape[0],
        weight.shape[1],
        input.shape[1],
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        activation,
    )

    return output

def triton_dgrad_act(d_output, input, weight, bias=None, activation="tanh"):
    # Prepare data
    d_output = d_output.contiguous()
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Initialize gradient tensors
    d_input = torch.empty_like(input)
    d_weight = torch.empty_like(weight)
    d_bias = torch.empty_like(bias) if bias is not None else None

    # Compute backward pass
    kernel_bwd[32, 32](
        d_output,
        d_input,
        weight,
        input,
        d_weight,
        d_bias,
        input.shape[0],
        weight.shape[1],
        input.shape[1],
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        activation,
    )

    return d_input, d_weight, d_bias

tanh_linear = lambda input, weight, bias=None: triton_linear_act(input, weight, bias, activation="tanh")
tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")
|end-of-document|>
<|system|>Document 2:
tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None: triton_dgrad_act(d_output, input, weight, bias, activation="tanh")

tanh_linear_grad = lambda d_output, input, weight, bias=None
