import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    dim: tl.constexpr
):
    # Get the program ID for the batch dimension
    batch_id = tl.program_id(0)
    
    # Load input tensor
    input_data = tl.load(input_ptr + batch_id * in_features)
    
    # Compute linear transformation
    linear_output = tl.dot(input_data, weight_ptr)  # y = xA^T
    if bias_ptr is not None:
        linear_output += tl.load(bias_ptr)
    
    # Compute log-softmax
    max_val = tl.max(linear_output, dim=dim)
    exp_vals = tl.exp(linear_output - max_val)
    log_softmax_output = linear_output - max_val - tl.log(tl.sum(exp_vals, dim=dim))
    
    # Store the result
    tl.store(output_ptr + batch_id * out_features, log_softmax_output)

def log_softmax_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, dim: int = -1, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""
    Applies a linear transformation to the input tensor followed by the log_softmax activation function.

    Args:
        input (Tensor): The input tensor of shape `(*, in_features)`.
        weight (Tensor): The weight matrix of shape `(out_features, in_features)`.
        bias (Tensor, optional): The optional bias tensor of shape `(out_features)`. Default: None.
        dim (int): The dimension along which log_softmax will be computed. Default: -1.
        dtype (Optional[torch.dtype]): The desired data type of the returned tensor. Default: None.

    Returns:
        Tensor: The result of applying the linear transformation followed by log_softmax.
    """
    # Reshape input tensor
    shape = input.shape
    input = input.view(-1, shape[-1])
    N, in_features = input.shape
    out_features = weight.shape[0]

    # Allocate output tensor
    output = input.new_empty(N, out_features, dtype=torch.float)

    # Launch the kernel
    log_softmax_linear_kernel[(N,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        in_features=in_features,
        out_features=out_features,
        dim=dim
    )

    # Cast to the desired dtype if specified
    if dtype is not None and dtype != torch.float:
        output = output.to(dtype)
    
    return output
