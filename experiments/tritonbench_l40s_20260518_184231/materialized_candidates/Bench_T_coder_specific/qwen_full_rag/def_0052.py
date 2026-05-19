import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def sum_std_kernel(
    input,
    output,
    mean_buffer,
    std_buffer,
    n_elements,
    cum_n_elements,
    input_element_stride,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    thread_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = thread_offsets < n_elements

    input_ptrs = input + thread_offsets * input_element_stride
    input_vals = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    if (
        tl.constexpr(isinstance(dim, int) and dim == 0)
        or tl.constexpr(isinstance(dim, tuple) and len(dim) == 1 and dim[0] == 0)
    ) or tl.constexpr(len(input.element_shape()) <= 1):
        current_sum = tl.sum(input_vals, axis=0)
        mean = current_sum / cum_n_elements
        diff_sq = tl.where(mask, (input_vals - mean) * (input_vals - mean), 0.0)
        variance = tl.sum(diff_sq, axis=0) / cum_n_elements
        std = tl.sqrt(variance)
    else:
        current_sum = tl.sum(input_vals, axis=0)
        mean = current_sum / N_ELEMENTS
        input_vals_centered = input_vals - mean
        diff_sq = input_vals_centered * input_vals_centered
        variance = tl.sum(diff_sq, axis=0) / (N_ELEMENTS - 1)
        std = tl.sqrt(variance)

    new_cum_n_elements = cum_n_elements + N_ELEMENTS

    tl.store(mean_buffer + pid, mean)
    tl.store(std_buffer + pid, std)
    tl.store(output + thread_offsets, input_vals)


@torch.inference_mode()
def sum_std(
    input: Tensor,
    dim=None,
    keepdim=False,
    *,
    dtype: Optional[torch.dtype] = None,
    correction: int = 1,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes the sum of elements in the input tensor along the specified dimension(s),
    followed by computing the standard deviation of each sum.
    .. note::
        This function will be renamed to ``var_mean`` in a future release.
    Args:
        input (Tensor): The input tensor
        dim (Optional[int]): The dimension to reduce. If ``None``, reduces all dims.
            Must be in range [-input.ndim, input.ndim).
        keepdim (bool): If True, retains reduced dims with length 1.
        dtype (Optional[torch.dtype]): Overrides the dtype of the output tensor.
        correction (int): Degrees of freedom adjustment for standard deviation calculation.
        out (Optional[Tensor]): The output tensor
    Returns:
        Tensor: The result of summing and then standardizing the input
    Example:
        >>> input = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
        >>> sum_std(input)
        tensor(1.1180, device='cuda')
        >>> sum_std(input, dim=0)
        tensor(1.1180, device='cuda')
        >>> sum_std(input, dim=-1)
        tensor(1.1180, device='cuda')
        >>> sum_std(input, dim=0, keepdim=True)
        tensor([[1.1180]], device='cuda')
        >>> sum_std(input, [0, -1])
        tensor(1.1180, device='cuda')
    """
    if dtype is None:
        dtype = input.dtype
    if out is not None:
        assert out.dtype == dtype
    # We reshape the input to a 2D tensor, perform the sum and std operations, and then restore its original shape.
    input_rank = input.ndim
    if dim is None or len(dim) == input_rank:
        input = input.contiguous()
        input_view_shape = (-1,)
        dim = 0
    else:
        assert isinstance(dim, int) or isinstance(dim, tuple)
        input, input_view_shape = make_contiguous(input, dim)
    assert input.is_cuda and input_view_shape is not None
    input_view = input.view(*input_view_shape)
    n_dims = len(input_view_shape)
    n_elements = input.numel() // input_view.size(0)
    block_size = triton.next_power_of_2(n_elements)
    num_blocks = min(65536, triton.cdiv(n_elements, block_size))
    block_size = triton.cdiv(n_elements, num_blocks)
    cum_n_elements = num_blocks * block_size
    num_full_blocks = (n_elements + block_size - 1) // block_size
    full_block_dtype = get_full_block_dtype(input.dtype)
    mean_buffer = torch.zeros((num_full_blocks,), dtype=full_block_dtype, device="cuda")
    std_buffer = torch.zeros((num_full_blocks,), dtype=full_block_dtype, device="cuda")

    if n_dims > 1:
        shape_prefix = input_view.shape[:-1]
        output_shape = list(shape_prefix) + [n_elements]
        output = (
            out.reshape(output_shape)
            if out is not None
            else torch.empty(*output_shape, dtype=dtype, device="cuda")
        )
        assert output.is_cuda
        output_view = output.narrow(dim, 0, n_elements)
    else:
        output = (
            out
            if out is not None
            else torch.empty(n_elements, dtype=dtype, device="cuda")
        )
        assert output.is_cuda
        output_view = output

    if n_elements <= 4096:
        BLOCK_SIZE = max_power_of_two(n_elements)
    elif n_elements <= 8192:
        BLOCK_SIZE = 4096
    else:
        BLOCK_SIZE = 8192

    BLOCK_SIZE = min(BLOCK_SIZE, 65536 // input.element_size())

    kernel_meta = get_kernel_meta()

    sum_std_kernel[(num_full_blocks,)](
        input_view,
        output_view,
        mean_buffer,
        std_buffer,
        n_elements,
        cum_n_elements,
        input_view.stride(-1),
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    means_out = mean_buffer[: num_blocks - 1].to(dtype, copy=True)
    stds_out = std_buffer[: num_blocks - 1].to(dtype, copy=True)
    final_mean = mean_buffer[-1].item()
    final_std = std_buffer[-1].item()

    def calc_final_mean_and_std(m, s, n):
        delta = m - final_mean
        alpha = n / (correction + n)
        final_mean += delta * alpha
        final_var = s * n / (correction + n) + delta * delta * alpha * final_mean
        final_std = math.sqrt(final_var)

    calc_final_mean_and_std(means_out, stds_out, torch.full_like(n_elements, num_blocks - 1))
    means_out[-1] = final_mean
    stds_out[-1] = final_std

    if n_dims > 1:
        means_out = means_out.reshape(shape_prefix + (n_elements,))
        stds_out = stds_out.reshape(shape_prefix + (n_elements,))
    return output
