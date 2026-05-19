import triton
import triton.language as tl

# Define the sigmoid kernel
@triton.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))

# Define the dropout kernel
@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))

# Define the main kernel for dropout_sigmoid_linear
@triton.jit
def dropout_sigmoid_linear_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr, size,
    drop_p, seed, inplace, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    # Load input and weight
    input = tl.load(input_ptr + offset, mask=mask)
    weight = tl.load(weight_ptr + offset, mask=mask)

    # Linear transformation: y = xW^T + b
    if inplace:
        output = input * weight.T + bias
    else:
        output = input @ weight.T + bias

    # Sigmoid activation
    output = sigmoid(output)

    # Dropout
    output = apply_dropout(output, drop_p, seed, offset)

    # Store the result
    tl.store(output_ptr + offset, output, mask=mask)

# Wrapper function
def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
    
    if not training or p == 0.0:
        output = input @ weight.T + bias
        output = torch.sigmoid(output)
        return output

    # Prepare output tensor
    output = input.new_zeros_like(input) if not inplace else input

    # Get grid size
    grid_size = (output.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    dropout_sigmoid_linear_kernel[grid_size, BLOCK_SIZE](
        input_ptr=input.contiguous().ptr,
        weight_ptr=weight.contiguous().ptr,
        output_ptr=output.contiguous().ptr,
        bias_ptr=bias.contiguous().ptr,
        size=output.numel(),
        drop_p=p,
        seed=torch.randint(0, 2**63, (1,), device=input.device).item(),
        inplace=inplace,
    )

    return output
