import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume
from .sigmoid_linear import sigmoid_linear, sigmoid_linear_inplace

@triton.jit
def dropout_sigmoid_linear_fwd(
    input_pointer, weight_pointer, bias_pointer, output_pointer,
    sigmoid_output_pointer, input_grad_pointer, weight_grad_pointer,
    bias_grad_pointer, size_n, size_c, size_f, num_blocks_m,
    num_blocks_k, num_warps, drop_p, seed, skip_bias: tl.constexpr,
    inplace: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ):
    """
    Randomly zeroes elements in the linear layer's input, then applies the
    linear layer, and stores the result.

    Args:
        input_pointer: Pointer to the linear layer's input.
            The input must be of shape [size_n, size_c].
        weight_pointer: Pointer to the linear layer's weight matrix T.
            The weight matrix must be of shape [size_c, size_f].
        bias_pointer: Pointer to the linear layer's bias vector.
            The vector must be of shape [size_f] or empty.
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size_n, size_f].
        sigmoid_output_pointer: Pointer to a container the intermediate result is written to.
            The container must be of shape [size_n, size_f].
        input_grad_pointer: Pointer to a container the input grad is written to.
            The container must be of shape [size_n, size_f].
        weight_grad_pointer: Pointer to a container the weight grad is written to.
            The container must be of shape [size_c, size_f].
        bias_grad_pointer: Pointer to a container the bias grad is written to.
            The container must be of shape [size_f].
        size_n: Number of examples.
        size_c: Number of channels.
        size_f: Number of features.
        num_blocks_m: Number of blocks of size BLOCK_SIZE_M.
        num_blocks_k: Number of blocks of size BLOCK_SIZE_K.
        num_warps: Number of warps.
        drop_p: Probability of dropping an element.
        seed: Seed for reproducibility.
        skip_bias: Whether to add bias to the output.
        inplace: If true uses the same memory for output and input.
        BLOCK_SIZE_M: Block size across the n dimension.
        BLOCK_SIZE_N: Block size across the c dimension.
        BLOCK_SIZE_K: Block size across the f dimension.
        MUL_ROUTED_WEIGHT: Multiplies the feature by the routed weight.
    """
    # This program processes BLOCK_SIZE_M rows and BLOCK_SIZE_K columns.
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    m_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    n_offset = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)).to(tl.int64)
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    offset_m = m_offset[:, None]
    offset_k = k_offset[None, :]
    input_offset = offset_m * size_c + offset_k
    weight_offset = n_offset * size_c + offset_k[None, :]
    mask_m = m_offset < size_n
    mask_k = n_offset < size_c
    mask = mask_m & mask_k

    input = tl.load(input_pointer + input_offset, mask=mask)
    weight = tl.load(weight_pointer + weight_offset, mask=mask)
    if skip_bias:
        bias = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    else:
        bias = tl.load(bias_pointer + n_offset, mask=n_offset < size_f)

    if inplace:
        output = input
    else:
        output = tl.load(output_pointer + input_offset, mask=mask)
    pre_act = tl.dot(input, weight, allow_tf32=False) + bias
    post_act = pre_act * (input > 0).to(tl.float32)
    dropout_mask = tl.rand(seed, offset_m) > drop_p
    output = tl.where(dropout_mask[:, None], post_act / (1 - drop_p), 0)
    tl.store(sigmoid_output_pointer + input_offset, post_act, mask=mask)
    tl.store(output_pointer + input_offset, output, mask=mask)

    if pid_m == 0:
        # Measures the sparsity after dropout.
        density = tl.sum(dropout_mask.to(tl.float32)) / (BLOCK_SIZE_M * BLOCK_SIZE_K)
        print(f'Dropout density {density}')

    if not inplace:
        input_grad = output
    else:
        input_grad = output + input
    tl.store(input_grad_pointer + input_offset, input_grad, mask=mask)
    grad_weight = input_grad
    tl.store(weight_grad_pointer + weight_offset, grad_weight, mask=mask)
    if not skip_bias:
        grad_bias = tl.sum(input_grad, axis=0)
        tl.store(bias_grad_pointer + n_offset, grad_bias, mask=n_offset < size_f)

@triton.jit
def dropout_sigmoid_linear_bwd(
    input_pointer, weight_pointer, sigmoid_output_pointer,
    input_grad_pointer, weight_grad_pointer, size_n, size_c,
    size_f, num_blocks_m, num_blocks_k, num_warps, drop_p,
    seed, inplace: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ):
    """
    Calculates the input and weight gradients of the linear layer with dropout.

    Args:
        input_pointer: Pointer to the linear layer's input.
            The input must be of shape [size_n, size_c].
        weight_pointer: Pointer to the linear layer's weight matrix T.
            The weight matrix must be of shape [size_c, size_f].
        sigmoid_output_pointer: Pointer to the linear layer's sigmoid output.
            The output must be of shape [size_n, size_f].
        input_grad_pointer: Pointer to a container the input grad is written to.
            The container must be of shape [size_n, size_f].
        weight_grad_pointer: Pointer to a container the weight grad is written to.
            The container must be of shape [size_c, size_f].
        size_n: Number of examples.
        size_c: Number of channels.
        size_f: Number of features.
        num_blocks_m: Number of blocks of size BLOCK_SIZE_M.
        num_blocks_k: Number of blocks of size BLOCK_SIZE_K.
        num_warps: Number of warps.
        drop_p: Probability of dropping an element.
        seed: Seed for reproducibility.
        inplace: If true uses the same memory for output and input.
        BLOCK_SIZE_M: Block size across the n dimension.
        BLOCK_SIZE_N: Block size across the c dimension.
        BLOCK_SIZE_K: Block size across the f dimension.
        MUL_ROUTED_WEIGHT: Multiplies the feature by the routed weight.
    """
    # This program processes BLOCK_SIZE_M rows and BLOCK_SIZE_K columns.
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    m_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    n_offset = (pid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    offset_m = m_offset[:, None]
    offset_k = k_offset[None, :]
    input_offset = offset_m * size_c + offset_k
    weight_offset = n_offset * size_c + offset_k[None, :]
    mask_m = m_offset < size_n
    mask_k = n_offset < size_c
    mask = mask_m & mask_k

    input = tl.load(input_pointer + input_offset, mask=mask)
    weight = tl.load(weight_pointer + weight_offset, mask=mask)
    output = tl.load(sigmoid_output_pointer + input_offset, mask=mask)
    input_grad = tl.load(input_grad_pointer + input_offset, mask=mask)
    grad_weight = input_grad * (output * (1 - output))
    tl.store(weight_grad_pointer + weight_offset, grad_weight, mask=mask)

    if not inplace:
        output_grad = input_grad
    else:
        output_grad = input_grad + input

    input_grad = output_grad * weight * (output * (1 - output) / (1 - drop_p))
    tl.store(input_grad_pointer + input_offset, input_grad, mask=mask)

def dropout_sigmoid_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
    p: float = 0.5, training: bool = True, inplace: bool = False,
    ) -> torch.Tensor:
    """
    Applies a linear transformation, a sigmoid activation, and dropout to the input.

    Args:
        input: Linear layer's input.
            The input can be of shape [size_n, size_c] or [size_c].
        weight: Linear layer's weight matrix T.
            The weight matrix should be of shape [size_c, size_f].
        bias: Linear layer's bias vector.
            The vector should be of shape [size_f].
        p: Probability of an element to be dropped.
        training: If
