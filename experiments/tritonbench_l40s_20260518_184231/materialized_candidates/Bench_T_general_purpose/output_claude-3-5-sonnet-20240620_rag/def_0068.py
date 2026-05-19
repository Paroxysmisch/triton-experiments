import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(input_ptr, other_ptr, out_ptr, alpha, dim, N, M):
    # Compute the index for the input tensor
    i = tl.program_id(0)
    # Load input and other tensors
    input_val = tl.load(input_ptr + i)
    other_val = tl.load(other_ptr + i) * alpha
    
    # Add the scaled other tensor to the input tensor
    result = input_val + other_val
    
    # Compute the mean along the specified dimension
    if dim is not None:
        result = tl.mean(result, dim)
    
    # Store the result in the output tensor
    tl.store(out_ptr + i, result)

def add_mean(input: torch.Tensor, other: torch.Tensor, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    r"""
    Adds the `other` tensor, scaled by `alpha`, to the `input` tensor and computes the mean value along the specified dimension.

    Args:
        input (Tensor): The input tensor.
        other (Tensor or Number): The tensor or number to add to input.
        dim (int or tuple of ints, optional): The dimension(s) to reduce. Default: None.
        alpha (Number, optional): The multiplier for other. Default: 1.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. Default: False.
        dtype (torch.dtype, optional): The desired data type of returned tensor. Default: None.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The result tensor after addition and mean computation.
    """
    
    # Ensure input and other are tensors
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, dtype=dtype)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=dtype)

    # Prepare output tensor
    if out is None:
        out_shape = input.shape
        if dim is not None:
            out_shape = list(out_shape)
            if isinstance(dim, int):
                out_shape[dim] = 1 if not keepdim else out_shape[dim]
            else:
                for d in dim:
                    out_shape[d] = 1 if not keepdim else out_shape[d]
        out = input.new_empty(out_shape, dtype=dtype)

    # Launch the kernel
    N = input.numel()
    M = other.numel()
    add_mean_kernel[(N,)](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        out_ptr=out.data_ptr(),
        alpha=alpha,
        dim=dim,
        N=N,
        M=M
    )
    
    return out
