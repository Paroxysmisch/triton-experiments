import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def normalize_kernel(
    input,
    output,
    norm_output,
    input_row_stride,
    n_cols,
    p_norm,
    eps_norm,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    sum_pow = tl.sum(xf**p_norm, 0)
    norm = tl.sqrt(sum_pow * float(1.0 / N_COLS)) + eps_norm
    norm_inv = 1.0 / norm
    normalized_x = xf * norm_inv

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, normalized_x, mask=offsets < n_cols)

    norm_out_ptr = norm_output + prog_id * input_row_stride
    tl.store(norm_out_ptr + offsets, norm, mask=offsets < n_cols)

@triton.jit
def cosine_similarity_kernel(
    x1_normalized,
    x2_normalized,
    output,
    x1_row_stride,
    x2_row_stride,
    n_cols,
    dim,
    eps_similarity,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    x1_ptr = x1_normalized + prog_id * x1_row_stride
    x1 = tl.load(x1_ptr + offsets, mask=offsets < n_cols)
    x2_ptr = x2_normalized + prog_id * x2_row_stride
    x2 = tl.load(x2_ptr + offsets, mask=offsets < n_cols)

    dot_product = tl.dot(x1, x2)
    norm1_ptr = x1_normalized + prog_id * x1_row_stride
    norm1 = tl.load(norm1_ptr + offsets, mask=offsets < n_cols)
    norm2_ptr = x2_normalized + prog_id * x2_row_stride
    norm2 = tl.load(norm2_ptr + offsets, mask=offsets < n_cols)

    max_norm1 = tl.max(norm1, eps_similarity)
    max_norm2 = tl.max(norm2, eps_similarity)
    similarity = dot_product / (max_norm1 * max_norm2)

    out_ptr = output + prog_id * x1_row_stride
    tl.store(out_ptr + offsets, similarity, mask=offsets < n_cols)

@torch.inference_mode()
def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> Tensor:
    """
    Computes the cosine similarity between two normalized input tensors `x1` and `x2`.

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.
        dim (int, optional): The dimension along which to perform normalization. Default is 1.
        eps_similarity (float, optional): A small value to avoid division by zero in similarity calculation. Default is 1e-8.
        p_norm (float, optional): The order of the L_p norm for normalization. Default is 2.
        eps_norm (float, optional): A small value to avoid division by zero in normalization. Default is 1e-12.

    Returns:
        Tensor: The output tensor containing the cosine similarity.
    """

    def _kernel_meta():
        device = x1.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    seq_len = x1.numel() // x1.size(dim)
    input_stride = x1.stride(dim)

    BLOCK_N = triton.next_power_of_2(seq_len)
    output = torch.empty_like(x1)

    # Prepare input tensors for normalization
    x1_normalized = torch.empty_like(x1)
    x2_normalized = torch.empty_like(x2)
    norm_x1 = torch.empty_like(x1)
    norm_x2 = torch.empty_like(x2)

    kernel_meta = _kernel_meta()
    grid = (seq_len,)
    
    normalize_kernel[grid](
        x1,
        x1_normalized,
        norm_x1,
        input_stride,
        seq_len,
        p_norm,
        eps_norm,
        seq_len,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    normalize_kernel[grid](
        x2,
        x2_normalized,
        norm_x2,
        input_stride,
        seq_len,
        p_norm,
        eps_norm,
        seq_len,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    cosine_similarity_kernel[grid](
        x1_normalized,
        x2_normalized,
        output,
        input_stride,
        input_stride,
        seq_len,
        dim,
        eps_similarity,
        seq_len,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        **kernel_meta,
    )

    return output
