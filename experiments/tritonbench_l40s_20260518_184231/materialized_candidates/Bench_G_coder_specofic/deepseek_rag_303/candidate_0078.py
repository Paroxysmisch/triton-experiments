import triton
import triton.language as tl

from torch._inductor.triton_heuristics import reduction, pointwise, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

from torch import empty_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor._triton_runtime as triton_runtime
from torch._inductor.ir import ReductionHint

from torch._inductor import triton_helpers
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_heuristics

from torch._inductor import triton_runtime

BLOCK_M: tl.constexpr = 16
BLOCK_N: tl.constexpr = 32
BLOCK_K: tl.constexpr = 32

SPLIT_K: tl.constexpr = 16

PRAGMAS = {
    "early_config_pragmas": [1, 4],  # for row-wise matrix multiplication
    "late_config_pragmas": [0, 2, 4] + list(range(8, 256 + BLOCK_M, 8)),
    "key_size_gcds": [1],
}

class _Int8MatMulRowwiseDequantize(triton.autotune.Tuple):
    pass

@triton.autotune(
    configs=[
        triton.autotune.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "SPLIT_K": 16,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.autotune.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "SPLIT_K": 16,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.autotune.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 8,
                "BLOCK_K": 32,
                "SPLIT_K": 16,
            },
            num_stages=4,
            num_warps=4,
        ),
        # more configs....
    ],
    key=[
        "has_bias",
        "bn_train",
        "is_cuda",
        "gated_activation",
        "K",
        "N",
    ],
    pragmas=PRAGMAS,
    meta={"signature": tuple},
)
@triton.jit
def _int8_matmul_rowwise_dequantize(
    state_x_ptr,
    state_w_ptr,
    input_ct,
    weights_ct,
    bias_ct,
    has_bias: tl.constexpr,
    bn_train: tl.constexpr,
    bn_channels,
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    touch_bn_stat,
    stride_x_group,
    stride_x_row,
    stride_x_col,
    stride_x_ct,
    stride_x_fr,
    stride_x_end,
    stride_weights_group,
    stride_weights_row,
    stride_weights_col,
    stride_weights_ct,
    stride_weights_fr,
    stride_weights_end,
    stride_bias_group,
    stride_bias_row,
    stride_bias_end,
    stride_out_group,
    stride_out_row,
    stride_out_col,
    stride_out_ct,
    stride_out_fr,
    stride_out_end,
    in_x_group_offset,
    in_x_row_offset,
    in_x_col_offset,
    in_x_ct_offset,
    in_x_fr_offset,
    in_weights_group_offset: tl.constexpr,
    in_weights_row_offset,
    in_weights_col_offset,
    in_weights_fr_offset,
    in_bias_group_offset,
    in_out_group_offset,
    in_out_row_offset,
    in_out_fr_offset,
    out_ct,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_IN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    STORE_MUL_RESULT: tl.constexpr,
    STORE_ADD_RESULT: tl.constexpr,
    ACTIVATION: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    grid = lambda META: [triton_runtime.cdiv(K, META["BLOCK_K"])]

@reduction(
    size_hints=[4, 4],  # could be inferred as 1, but guessing this is a good heuristic
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={
        "signature": torch.empty_like,
        "device": 0,
        "constants": {"out_ct": 0},  # this is wrong but we need to call our autotuned kernel
        "configs": [
            instance_descriptor(divisible_by_16={"K": 1, "N": 1, "M": 1}),
            instance_descriptor(equal_to_1={"M": 1}),
        ],
    },
)
@pointwise(
    filename=__file__,
    meta={"signature": torch.empty_like},
)
@extern_kernels(meta={"signature": empty_strided})
@persistent_reduction(
    filename=__file__,
    reduction_hint=ReductionHint.INNER,
    meta={
        "signature": torch.empty_like,
        "device": 0,
        "constants": {"out_ct": 0},
        "configs": [instance_descriptor(divisible_by_16={"N": 1, "M": 1},
    },
)
def int8_matmul_rowwise_dequantize(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    bn_weight: Optional[torch.Tensor] = None,
    bn_bias: Optional[torch.Tensor] = None,
    bn_running_mean: Optional[torch.Tensor] = None,
    bn_running_var: Optional[torch.Tensor] = None,
    touch_bn_stat: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    stable: bool = False,
    gated_activation: bool = False,
):
    if isinstance(input, torch.Tensor) and stable:
        input = input.mul(0.5).add(0.5).round().to(torch.int32)
    else:
        input = input.round().to(torch.int32)
    if isinstance(weights, torch.Tensor) and stable:
        weights = weights.mul(0.5).add(0.5
