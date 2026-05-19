import torch
import triton
import triton.language as tl

@triton.jit
def _smelu_kernel_forward(
        input_pointer,
        beta: float,
        output_pointer,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel SmeLU forward """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_pointer + offsets, mask=mask)
    output = tl.where(x >= beta, x, 0.)
    output = tl.where(tl.abs(x) <= beta, ((x + beta) * (x + beta)) / (4. * beta), output)
    # Write-back output
    tl.store(output_pointer + offsets, output, mask=mask)

def _smelu_triton_forward(
        input: torch.Tensor,
        beta: float = 2.
) -> torch.Tensor:
    """
    Wrapper function for SmeLU forward triton kernel
    :param input (torch.Tensor): Input tensor of any shape
    :param beta (float): Beta value of SmeLU
    :return (torch.Tensor): Activation of SmeLU
    """
    # Init output tensor
    output: torch.Tensor = torch.empty_like(input)
    # Make input contiguous if needed
    if not input.is_contiguous():
        input = input.contiguous()
    # Get number of elements in input
    number_of_elements: int = input.numel()
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(number_of_elements, meta['BLOCK_SIZE']),)
    _smelu_kernel_forward[grid](input, beta, output, number_of_elements, BLOCK_SIZE=1024)
    return output

@triton.jit
def _smelu_kernel_backward(
        input_pointer,
        beta: float,
        output_pointer,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel SmeLU backward """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_pointer + offsets, mask=mask)
    gradient = tl.where(x >= beta, 1., 0.)
    gradient = tl.where(tl.abs(x) <= beta, 0.5 * (x + beta) / beta, gradient)
    # Write-back output
    tl.store(output_pointer + offsets, gradient, mask=mask)

def _smelu_triton_backward(
        input: torch.Tensor,
        beta: float = 2.
) -> torch.Tensor:
    """
    Wrapper function for SmeLU backward triton kernel
    :param input (torch.Tensor): Input tensor of any shape
    :param beta (float): Beta value of SmeLU
    :return (torch.Tensor): Gradient of SmeLU
    """
    # Init output tensor
    output: torch.Tensor = torch.empty_like(input)
    # Make input contiguous if needed
    if not input.is_contiguous():
        input = input.contiguous()
    # Get number of elements in input
    number_of_elements: int = input.numel()
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(number_of_elements, meta['BLOCK_SIZE']),)
    _smelu_kernel_backward[grid](input, beta, output, number_of_elements, BLOCK_SIZE=1024)
    return output
