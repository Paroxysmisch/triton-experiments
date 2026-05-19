import torch
import triton
import triton.language as tl
from torch import Tensor
from flag_gems.utils.shape_utils import volume
from flag_gems.ops.triton.activations import gelu_functor

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "P"],
)
@triton.heuristics({"HAS_BIAS": lambda args: args["bias"] is not None})
@triton.jit
def _fused_bmm_rmsnorm_gelu_dropout_kernel(
    a_ptr,
    b_ptr,
    pre_lin_ptr,
    rms_weight_ptr,
    post_lin_ptr,
    bias_ptr,
    mean_ptr,
    inv_std_ptr,
    output_ptr,
    dropout_ptr,
    stride_a_batch,
    stride_a_n,
    stride_a_m,
    stride_b_batch,
    stride_b_m,
    stride_b_p,
    stride_pre_batch,
    stride_pre_n,
    stride_pre_p,
    stride_post_batch,
    stride_post_n,
    stride_post_p,
    stride_output_batch,
    stride_output_n,
    stride_output_p,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
    approximate_gelu: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    DO_BMM: tl.constexpr,
    OUTPUT_LAST_DIM_ONE: tl.constexpr,
    STORE_DROPOUT: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = tl.program_id(axis=0) % num_pid_m
    pid_b = tl.program_id(axis=0) // num_pid_m

    a_ptr += pid_batch * stride_a_batch + pid_n * BLOCK_SIZE_M * stride_a_n
    b_ptr += pid_batch * stride_b_batch + pid_n * BLOCK_SIZE_M * stride_b_m
    pre_lin_ptr += (
        pid_b * stride_pre_batch
        + pid_n * BLOCK_SIZE_M * stride_pre_n
        + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_pre_n + tl.arange(0, P)[None, :] * stride_pre_p)
    )
    rms_weight_ptr += pid_b * BLOCK_SIZE_M * stride_pre_n + tl.arange(0, BLOCK_SIZE_M) * stride_pre_n
    post_lin_ptr += (
        pid_b * stride_post_batch
        + pid_n * BLOCK_SIZE_M * stride_post_n
        + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_post_n + tl.arange(0, P)[None, :] * stride_post_p)
    )
    output_ptr += (
        pid_batch * stride_output_batch
        + pid_n * BLOCK_SIZE_M * stride_output_n
        + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_output_n + tl.arange(0, P)[None, :] * stride_output_p)
    )

    if STORE_DROPOUT:
        dropout_ptr += (
            pid_batch * stride_output_batch
            + pid_n * BLOCK_SIZE_M * stride_output_n
            + (tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_output_n + tl.arange(0, P)[None, :] * stride_output_p)
        )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_p = tl.arange(0, P)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_n + offs_p[None, :] * stride_a_m)
    b_ptrs = b_ptr + (offs_p[:, None] * stride_b_m + offs_n[None, :] * stride_b_p)

    accumulator = tl.zeros((BLOCK_SIZE_M, P), dtype=tl.float32)

    if DO_BMM:
        for m in range(0, tl.cdiv(N, BLOCK_SIZE_M)):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (offs_p[None, :] < P), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_p[:, None] < N) & (offs_n[None, :] < P), other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_SIZE_M * stride_a_n
            b_ptrs += BLOCK_SIZE_M * stride_b_m

    else:
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < N) & (offs_p[None, :] < P), other=0.0)
        accumulator += a

    pre_lin = tl.load(pre_lin_ptr, mask=(offs_m[:, None] < N) & (offs_p[None, :] < P), other=0.0)
    if IS_RMS_NORM:
        var = tl.sum(accumulator * accumulator, axis=1)
    else:
        var = tl.sum(accumulator * accumulator, axis=0)

    inv_std = tl.rsqrt(var / N + 1e-5)
    gamma = tl.load(rms_weight_ptr)
    accumulator = accumulator * (inv_std[:, None] * gamma[None, :])

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_p, mask=offs_p < P, other=0.0)
        accumulator += bias[None, :]

    load_output = True
    if OUTPUT_LAST_DIM_ONE:
        if P > 1:
            load_output = False
        set_last_dim_one = True

    else:
        set_last_dim_one = False

    post_accumulator = tl.load(post_lin_ptr, mask=(offs_m[:, None] < N) & (offs_p[None, :] < P), other=0.0)
    accumulator += post_accumulator

    if approximate_gelu == "tanh":
        output = gelu_functor.approximate_gelu_tanh(accumulator)

    elif approximate_gelu == "none":
        output = gelu_functor.gelu(accumulator, deterministic=DETERMINISTIC)

    msk = (offs_m[:, None] < N) & (offs_p[None, :] < P)

    if load_output:
        tl.store(output_ptr, output, mask=msk)

    if STORE_DROPOUT:
        dropout = tl.rand(tl.float32)
        tl.store(dropout_ptr, dropout, mask=msk)


def fused_bmm_rmsnorm_gelu_dropout(
    input1: Tensor,
    input2: Tensor,
    normalized_shape: Union[int, List[int], Size],
    *,
    dropout_p: float = 0.1,
    eps: float = 1e-5,
    traininig: bool = True,
    approximate: str = "none",
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    pre_linear_weight: Optional[Tensor] = None,
    post_linear_weight: Optional[Tensor] = None,
    deterministic: bool = False,
    store_dropout: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    _check_bmm_shapes(input1, input2)
    input_device = input1.device
    if input_device.type != "cuda":
        raise RuntimeError("Fused BMMLayerNorm only supports CUDA device")

    input_dtype = input1.dtype
    if input_dtype not in [torch.float16, torch.bfloat16]:
        raise RuntimeError("BMMLayerNorm only supports FP16 and BF16")

    if input1.dim() != 3:
        raise RuntimeError("Expected input1 to be a 3D tensor")

    if input2.dim() != 3:
        raise RuntimeError("Expected input2 to be a 3D tensor")

    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    elif isinstance(normalized_shape, list):
        assert len(normalized_shape) == 1, "List length must be 1"
        normalized_shape = [normalized_shape[0]]

    elif isinstance(normalized_shape, Size):
        normalized_shape = [normalized_shape[0]]

    else:
        raise RuntimeError("Unknown type for normalized_shape")

    if normalized_shape[0] <= 0:
        raise RuntimeError("normalized_shape must be greater than 0")

    if input1.size(2) != normalized_shape[0]:
        raise RuntimeError("Size of tensor along the feature dimension must be same as feature dim size")

    if input2.size(2) != normalized_shape[0]:
        raise RuntimeError("Size of tensor along the feature dimension must be same as feature dim size")

    if approximate != "tanh" and approximate != "none":
        raise ValueError(f"Invalid approximate value: {approximate}")

    if out is not None and out.shape != input1.shape:
        raise RuntimeError("The shape of out and input1 must be the same")

    if out is None:
        out = torch.empty_like(input1)

    else:
        assert out.is_contiguous()

    if pre_linear_weight is not None:
        pre_linear_weight = pre_linear_weight.expand(len(input1), -1
