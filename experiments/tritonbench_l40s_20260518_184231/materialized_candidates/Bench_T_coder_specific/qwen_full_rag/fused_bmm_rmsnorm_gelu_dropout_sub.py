import torch
import triton
import triton.language as tl
from .gelu import gelu_functor
from .dropout import dropout
from ..utils import calculate_settings

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    other_ptr,
    scale_ptr,
    mean_square_ptr,
    rms_ptr,
    N,
    M,
    P,
    sequence_stride,
    hidden_stride,
    epilogue_stride,
    K,
    dropout_probability,
    eps,
    training,
    approximate_gelu,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTER_OUTPUT: tl.constexpr,
    SAVE_RMS_AND_MEAN_VAR: tl.constexpr,
):
    """
    Implements the fused operation of batch matrix multiplication, RMS normalization,
    GELU activation, dropout, and subtraction using Triton kernels.
    """

    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    cur_batch = pid_b // N
    cur_seq = pid_b % N

    input_batch_offset = cur_batch * sequence_stride
    input_hidden_offset = (cur_seq * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * hidden_stride
    input_block_ptr = (
        input_ptr + input_batch_offset + input_hidden_offset + tl.arange(0, BLOCK_SIZE_K)[None, :]
    )

    rmse_batch_offset = cur_batch * N * P
    rmse_task_offset = tl.arange(0, BLOCK_SIZE_N) * P
    mean_square_ptr = mean_square_ptr + rmse_batch_offset + rmse_task_offset[:, None]
    rms_ptr = rms_ptr + rmse_batch_offset + rmse_task_offset[:, None]

    k_stitch = tl.arange(0, BLOCK_SIZE_K)
    input_mask = k_stitch[None, :] < M
    input_tile_offset = k_stitch * hidden_stride
    input_a_ptr = input_ptr + input_batch_offset + input_hidden_offset + input_tile_offset[None, :]

    accumulator_dtype = tl.float32
    accumulators = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    loop_k_iterations = K // BLOCK_SIZE_K
    for k in range(loop_k_iterations):
        tiled_A = tl.load(input_a_ptr, mask=input_mask, other=0.0)
        tiled_B = tl.load(weight_ptr + k * hidden_stride + tl.arange(0, BLOCK_SIZE_K))
        accumulators += tl.dot(tiled_A, tiled_B, allow_tf32=True)
        input_a_ptr += BLOCK_SIZE_K * hidden_stride

    if SAVE_RMS_AND_MEAN_VAR:
        mean_squares = tl.sum(accumulators * accumulators, axis=1) / K
        tl.store(mean_square_ptr, mean_squares)
        tl.debug_barrier()
        rms_values = tl.rsqrt(mean_squares + eps)
        tl.store(rms_ptr, rms_values)

    if MUL_ROUTER_OUTPUT:
        router_out_d_ptr = scale_ptr + tl.arange(0, BLOCK_SIZE_N)
        accumulators *= tl.expand_dims(tl.load(router_out_d_ptr), 0)

    bias_offset = cur_batch * sequence_stride + cur_seq * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    other_block_ptr = other_ptr + bias_offset + tl.arange(0, BLOCK_SIZE_N)[None, :]
    other_mask = tl.arange(0, BLOCK_SIZE_N)[None, :] < P
    b_ptr = other_block_ptr
    b = tl.load(b_ptr, mask=other_mask, other=float("-inf")).to(accumulator_dtype)

    accumulators = gelu_functor(accumulators, approximate=approximate_gelu)
    if dropout_probability > 0.0:
        keep_prob = 1.0 - dropout_probability
        if training:
            accumulators = dropout(accumulators, keep_prob)
        else:
            accumulators *= keep_prob
    accumulators -= b

    output_bias_offset = cur_batch * sequence_stride + cur_seq * BLOCK_SIZE_M +
    tl.arange(0, BLOCK_SIZE_M)
    output_block_ptr = (output_ptr + output_bias_offset +
                        tl.arange(0, BLOCK_SIZE_N)[None, :])
    output_mask = tl.arange(0, BLOCK_SIZE_N)[None, :] < P
    c_ptr = output_block_ptr
    tiled_C = accumulators.to(tl.float16)
    tl.store(c_ptr, tiled_C, mask=output_mask)


def fused_bmm_rmsnorm_gelu_dropout_add_other(
    input1: torch.Tensor,
    input2: torch.Tensor,
    other: torch.Tensor,
    normalized_shape: int,
    dropout_p: float = 0.5,
    training: bool = True,
    approximate_gelu: str = "none",
    eps: float = 1e-5,
    save_rms_and_mean_var: bool = False,
    mul_router_output: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Arguments:
        input1 (Tensor): First input tensor for batch matrix multiplication.
            Expected shape is (batch_size, seq_len, in_features).
        input2 (Tensor): Second input tensor for batch matrix multiplication.
            Expected shape is (batch_size, in_features, out_features).
        other (Tensor): Subtracted from the final result.
            Expected shape is (seq_len, out_features) or (batch_size, seq_len, out_features).
        normalized_shape (int): Shape over which RMS normalization is applied.
        dropout_p (float, optional): Probability of an element to be zeroed in the dropout layer.
            Default: 0.5.
        training (bool, optional): Apply dropout if true. Default: True.
        approximate_gelu (str, optional): Can be 'none' or 'tanh'.
            The approximation to use for GELU. Default: 'none'.
        eps (float, optional): Value added to the denominator for numerical stability in RMS
            normalization. Default: 1e-5.
        save_rms_and_mean_var (bool, optional): Save RMS and mean/variance to their respective
            outputs. Default: False.
        mul_router_output (bool, optional): Multiply the router output. Default: False.
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    """

    assert input1.dim() == 3, "Input 1 must be 3-dimensional"
    assert input2.dim() == 3, "Input 2 must be 3-dimensional"
    assert other.dim() == 2 or other.dim() == 3, "Other must be 2- or 3-dimensional"

    batch_size, seq_len, in_features = input1.shape
    _, in_features_check, out_features = input2.shape
    assert in_features == in_features_check, "Infeatures must match"
    assert normalized_shape == out_features, "Normalized shape must equal out features"
    assert (
        other.shape == [seq_len, out_features]
        or other.shape == [batch_size, seq_len, out_features]
    ), "Incorrect other tensor shape"

    if other.shape[0] != batch_size:
        other = other.unsqueeze(0).expand(batch_size, seq_len, out_features)

    if out is None:
        output = torch.empty(
            (batch_size, seq_len, out_features),
            device=input1.device,
            dtype=input1.dtype,
        )
    else:
        output = out

    if save_rms_and_mean_var:
        rms = torch.empty(
            (batch_size, seq_len, out_features),
            device=input1.device,
            dtype=torch.float32,
        )
        mean_square = torch.empty(
            (batch_size, out_features),
            device=input1.device,
            dtype=torch.float32,
        )
        epilogue_stride = output.stride(0) * output.stride(1)
    else:
        rms = None
        mean_square = None
        epilogue_stride = 0

    weight_strides = input2.stride()
    hidden_stride = weight_strides[weight_strides.index(max(weight_strides))]
    K = max(input2.size(i) // hidden_stride for i in range(len(weight_strides)))

    grid = lambda META: (
        triton.cdiv(seq_len, META["BLOCK_SIZE_M"]),
        batch_size * triton.cdiv(seq_len, META["BLOCK_SIZE_M"]),
    )
    settings = calculate_settings(K, min_num_warps=8)

    with torch.cuda.device(input1.device):
        fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
            input1,
            input2,
            output,
            other,
            None if scale_ptr is None else scale_ptr,
            mean_square,
            rms,
            seq_len,
            in_features,
            out_features,
            input1.stride(0),
            input1.stride(1),
            output.stride(0),
            K,
            dropout_p,
            eps,
            training,
            approximate_gelu,
            MUL_ROUTER_OUTPUT=mul_router_output,
            SAVE_RMS_AND_MEAN_VAR=save_rms_and_mean_var,
            **settings,
        )
    return output, rms, mean_square if save_rms_and_mean_var else output
