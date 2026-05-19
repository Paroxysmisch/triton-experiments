import torch
import triton
import triton.language as tl


@triton.jit
def tanh_linear_jit(input, weight, bias, out):
    out_orig_strides = out.stride()
    out = out.view(-1, out.shape[-1])
    m, n = out.shape
    k = input.shape[-1]
    input = input.reshape(m, k)
    # out = tl.zeros((m, n), dtype=tl.float32)
    # input batches are along dim 0, and we keep processing these batches independently
    tile_size = 64
    num_tiles = triton.cdiv(k, tile_size)
    for i in range(num_tiles):
        cols = i * tile_size + tl.arange(0, tile_size)
        mask = cols < k
        x = tl.load(input + i * tile_size + cols, mask=mask[None, :]).to(tl.float32)
        w = tl.load(weight + cols, mask=mask).to(tl.float32)
        x = tl.dot(x, w, allow_tf32=False)
        if bias is not None:
            b = tl.load(bias + cols, mask=mask).to(tl.float32)
            x += b
        x = tl.libdevice.tanh(x)
        tl.store(out + i * tile_size + cols, x, mask=mask)
    return out.view(*out_orig_strides)


def tanh_linear(input, weight, bias=None) -> torch.Tensor:
    """
    Linearly transforms the input and applies the Tanh activation function.
    Args:
        input (Tensor): The input tensor of shape `(*, in_features)`.
            `*` can be zero or more batch dimensions.
        weight (Tensor): The weight matrix of shape `(out_features, in_features)`.
        bias (Tensor, optional): The optional bias tensor of shape `(out_features)`.
            Default: None.
    Returns:
        Tensor: The output tensor of shape `(*, out_features)`.
    """
    input_dim = input.shape[-1]
    output_dim = weight.shape[0]
    if input_dim != weight.shape[1]:
        raise ValueError(
            f"Incompatible dimensions in between input and weight ({input_dim} != {weight.shape[1]})"
        )
    if bias is not None and bias.shape[0] != output_dim:
        raise ValueError(
            f"Incompatible dimensions in between weight and bias ({output_dim} != {bias.shape[0]})"
        )

    if len(input.shape) == 1:
        input = input.unsqueeze(0)

    output = torch.empty(
        (*input.shape[:-1], output_dim), dtype=input.dtype, device=input.device
    )
    # TODO tune this
    grid = lambda meta: (triton.cdiv(input.numel(), meta["BLOCK_SIZE"]),)
    # call the JIT function
    tanh_linear_jit[grid](input, weight, bias, output)
    return output
