import torch
import triton
import triton.language as tl

@triton.jit
def _fused_cross_entropy_log_softmax(
    input, target, weight, ignore_index, label_smoothing, N, C, logit_scale, ONE, reduction: tl.constexpr
):
    # Apply logit scaling
    input *= logit_scale

    # Clamp input for numerical stability
    input = tl.where(input <= 100.0, input, 100.0)

    # Compute log sumexp
    row_max = tl.max(input, 1)
    row_max_outside = tl.max(input, 1)[:, None]
    numerator = tl.exp(input - row_max_outside)
    denominator = tl.sum(numerator, 1)[:, None]

    # Log softmax
    input = -tl.log(denominator) - row_max + tl.log(numerator)

    # Cross entropy loss
    target = tl.where(target != ignore_index, target, -1)
    idx = tl.arange(0, N) * C + target
    input = tl.sum(tl.load(input + idx) * weight[target] * (ONE - label_smoothing), 0) / N

    if reduction == "mean":
        pass

    elif reduction == "sum":
        input = -input

    return input

def fused_cross_entropy_log_softmax(input, target, dim=1, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0) -> torch.Tensor:
    assert (
        reduction in ["none", "mean", "sum"]
    ), f"Invalid reduction method {reduction}. Expected one of {'none', 'mean', 'sum'}"

    assert 0 <= label_smoothing < 1, f"Invalid smoothing factor {label_smoothing}"

    if weight is not None:
        assert isinstance(weight, torch.Tensor) and weight.ndim == 1, f"Invalid weight shape {weight.shape}"
        assert weight.device.type == input.device.type and weight.device.index == input.device.index, (
            f"Weight and input must be on the same device; got {weight.device} and {input.device} respectively"
        )
        assert len(weight) == num_classes := input.shape[dim], f"Incompatible number of classes: {len(weight)} != {num_classes}"
        weight = weight.contiguous()
    else:
        weight = torch.ones(input.shape[input.ndim - 1], dtype=input.dtype, device=input.device)

    assert target.dtype in (torch.int64, torch.int32)

    if target.ndim != input.ndim - 1:
        raise ValueError("The shape of target must be broadcastable with input excluding the last dimension")

    if input.ndim < 2:
        raise ValueError("Input tensor must be at least 2D")

    logit_scale = 1.0

    grid_fn = lambda meta: (triton.cdiv(input.shape[0], meta["BLOCK_SIZE"]),)

    loss_dtype = input.dtype
    if loss_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        loss_dtype = torch.float32

    if input.requires_grad:

        @contiguous
        @jit
        def fused_cross_entropy_log_softmax_fwd(input, target, weight, ignore_index, label_smoothing):
            B, V = input.shape
            block_size = 4
            scale = math.ceil(math.sqrt(V) / block_size)
            if scale > 1:
                inputs = F.pad(input, (0, scale * block_size - input.size(-1)), mode="constant", value=0).view(B, scale, block_size)
                target = target % V
                result = tl.zeros([B,], dtype=loss_dtype)
                for i in range(scale):
                    mask = (i * block_size) + tl.arange(0, block_size)
                    x = tl.max(inputs[:, i, :], 1)[:, None]
                    y = tl.where(mask[:, None] == target[:, None], 1.0, 0.0)
                    z = inputs[:, i, :] - x
                    w = tl.exp(z)
                    s = tl.sum(w, 1)
                    o = -(x.squeeze() + tl.log(s))
                    result += tl.sum(o * y, 1)
                result /= B
                return result.to(loss_dtype)
            else:
                return _fused_cross_entropy_log_softmax(
                    input,
                    target,
                    weight,
                    ignore_index,
                    label_smoothing,
                    input.shape[0],
                    input.shape[1],
                    logit_scale,
                    torch.tensor(1.0 / input.shape[0]).to(loss_dtype),
                    reduction=reduction,
                )

        return fused_cross_entropy_log_softmax_fwd(input, target, weight, ignore_index, label_smoothing)
    else:
        with torch.cuda.device(input.device):

            @contiguous
            @jit
            def fused_cross_entropy_log_softmax_inference(input, target, weight, ignore_index, label_smoothing):
                B, V = input.shape
                block_size = 4
                scale = math.ceil(math.sqrt(V) / block_size)
                if scale > 1:
                    inputs = F.pad(input, (0, scale * block_size - input.size(-1)), mode="constant", value=0).view(B, scale, block_size)
                    target = target % V
                    result = tl.zeros([B,], dtype=loss_dtype)
                    for i in range(scale):
                        mask = (i * block_size) + tl.arange(0, block_size)
                        x = tl.max(inputs[:, i, :], 1)[:, None]
                        y = tl.where(mask[:, None] == target[:, None], 1.0, 0.0)
                        z = inputs[:, i, :] - x
                        w = tl.exp(z)
                        s = tl.sum(w, 1)
                        o = -(x.squeeze() + tl.log(s))
                        result += tl.sum(o * y, 1)
                    result /= B
                    return result.to(loss_dtype)
                else:
                    return _fused_cross_entropy_log_softmax(
                        input,
                        target,
                        weight,
                        ignore_index,
                        label_smoothing,
                        input.shape[0],
                        input.shape[1],
                        logit_scale,
                        torch.tensor(1.0).to(loss_dtype),
                        reduction=reduction,
                    )

            return fused_cross_entropy_log_softmax_inference(input, target, weight, ignore_index, label_smoothing)
