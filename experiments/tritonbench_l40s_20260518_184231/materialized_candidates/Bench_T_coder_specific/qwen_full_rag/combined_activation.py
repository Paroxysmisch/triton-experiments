import torch
import triton
import triton.language as tl


@triton.jit
def combined_activations_triton(
    input, 
    weight1, 
    weight2, 
    bias
):
    """Triton implementation of combined activations.

    Args:
        input (torch.Tensor): Input tensor of shape (*, N, D_in).
        weight1 (torch.Tensor): First weight matrix of shape (D_in, D_out).
        weight2 (torch.Tensor): Second weight matrix of shape (D_out,).
        bias (torch.Tensor): Bias vector of shape (D_out,).

    Returns:
        torch.Tensor: Output tensor of shape (*, N, D_out).
    """
    # Get the leading dimensions of the input tensor
    ld = input.shape[:-2]
    # Reshape the input tensor to remove leading dimensions
    input = input.reshape(-1, input.shape[-2], input.shape[-1])
    output = tl.zeros(input.shape, dtype=input.dtype)
    # Define the number of blocks and the size of each block
    blk = 64
    n_blks = (input.shape[2] + blk - 1) // blk
    # Iterate over the blocks and compute the activation function
    for i in range(n_blks):
        tmp = tl.dot(input, weight1)
        sig = tl.sigmoid(tmp)
        tan = tl.tanh(tmp)
        wot = tan * sig
        res = tl.dot(wot, weight2) + bias
        output = tl.where(i == 0, res, output + res)
    # Reshape the output tensor to have the same leading dimensions as the input
    output = output.reshape(*ld, output.shape[-2], output.shape[-1])
    return output


def combined_activations_wrapper(input, weight1, weight2, bias):
    """Python wrapper for calling the Triton kernel."""
    # Check that the input tensors have the correct dimensions
    assert input.dim() >= 2
    assert weight1.shape[1] == input.shape[-1]
    assert weight2.shape[0] == weight1.shape[0]
    assert bias.shape[0] == weight2.shape[0]
    # Call the Triton kernel
    return combined_activations_triton(input, weight1, weight2, bias)
