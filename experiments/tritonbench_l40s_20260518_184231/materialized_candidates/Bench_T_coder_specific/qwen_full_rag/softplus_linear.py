import torch
import triton
import triton.language as tl
from ssd.bi.softplus import softplus

@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_row_stride,
    weight_row_stride,
    weight_col_stride,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """
    Kernel for computing Out = activation.Linear(Input).
    Input has shape (M, K), weight has shape (K, N) and bias has shape (N,)
    Output has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    input_block_ptr = (
        input_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
        * input_row_stride
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )

    weight_block_ptr = (
        weight_ptr
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None]
        * weight_col_stride
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
        * weight_row_stride
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        input = tl.load(
            input_block_ptr,
            mask=(k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K - k,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_block_ptr,
            mask=(k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K - k,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.dot(input, weight)
        input_block_ptr += BLOCK_SIZE_K
        weight_block_ptr += BLOCK_SIZE_K

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_N)[:, None], mask=True).to(
            tl.float32
        )
        accumulator += bias

    output = accumulator.to(tl.float16)

    if SOFTPLUS_BETA != 1 or SOFTPLUS_THRESHOLD != 20:
        output = softplus(output, beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD)

    output_block_ptr = (
        output_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
        * input_row_stride
        + tl.arange(0, BLOCK_SIZE_N)[None, :]
    )

    tl.store(output_block_ptr, output, mask=True)


def softplus_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    beta: int = 1,
    threshold: int = 20,
) -> torch.Tensor:
    """
    Compute the linear layer with Softplus activation function.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K].
        weight (torch.Tensor): Weight tensor of shape [K, N].
        bias (torch.Tensor, optional): Bias tensor of shape [N]. Defaults to None.
        beta (int, optional): Beta value for SoftPlus. Defaults to 1.
        threshold (int, optional): Threshold value for SoftPlus. Defaults to 20.

    Returns:
        torch.Tensor: Output tensor.
    """

    assert input.dtype == weight.dtype, f"Input and weight must have the same dtype, got {input.dtype} and {weight.dtype}"
    assert input.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"

    if bias is not None:
        assert input.size(1) == weight.size(0), "Incompatible dimensions"
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.size(0) == weight.size(1), "Incompatible dimensions"
    else:
        assert input.size(1) == weight.size(0), "Incompatible dimensions"

    assert (
        2 ** tl.next_power_of_2(weight.shape[1]) >= weight.shape[1]
    ), "Number of output features must be a power of two"

    M, K = input.shape
    N, K = weight.shape

    output = torch.empty((M, N), device=input.device, dtype=torch.float16)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    softplus_linear_kernel[grid](
        input,
        weight,
        bias,
        output,
        input.stride(0),
        weight.stride(0),
        weight.stride(1),
        M,
        N,
        K,
        SOFTPLUS_BETA=beta,
        SOFTPLUS_THRESHOLD=threshold,
    )

    return output
