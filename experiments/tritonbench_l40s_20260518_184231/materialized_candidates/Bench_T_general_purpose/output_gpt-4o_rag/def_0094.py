import torch
import triton
import triton.language as tl

@triton.jit
def linear_sigmoid_dropout_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, 
    size, in_features, out_features, 
    drop_p, seed, training,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate program IDs
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    # Load input and weights
    input = tl.load(input_ptr + offset, mask=mask)
    weight = tl.load(weight_ptr + offset % out_features, mask=mask)
    bias = tl.load(bias_ptr + offset % out_features, mask=mask) if bias_ptr else 0.0

    # Linear transformation
    linear_output = tl.dot(input, weight) + bias

    # Sigmoid activation
    sigmoid_output = 1 / (1 + tl.exp(-linear_output))

    # Apply dropout if training
    if training:
        random = tl.rand(seed, offset)
        dropout_mask = random >= drop_p
        sigmoid_output = tl.where(dropout_mask, sigmoid_output / (1 - drop_p), 0)

    # Store result
    tl.store(output_ptr + offset, sigmoid_output, mask=mask)

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Check dimensions
    assert input.size(-1) == weight.size(-1), "Input features must match weight's in_features"
    assert weight.size(0) == (bias.size(0) if bias is not None else weight.size(0)), "Output features must match bias size"

    # Prepare output tensor
    output = input if inplace else torch.empty_like(input)

    # Calculate sizes
    size = input.numel()
    in_features = input.size(-1)
    out_features = weight.size(0)

    # Allocate device memory for bias if necessary
    bias_ptr = bias.data_ptr() if bias is not None else 0

    # Launch kernel
    BLOCK_SIZE = 128  # Example block size
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    linear_sigmoid_dropout_kernel[grid](
        input.data_ptr(), weight.data_ptr(), bias_ptr, output.data_ptr(),
        size, in_features, out_features,
        p, torch.randint(0, 2**31 - 1, (1,)).item(), training,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
