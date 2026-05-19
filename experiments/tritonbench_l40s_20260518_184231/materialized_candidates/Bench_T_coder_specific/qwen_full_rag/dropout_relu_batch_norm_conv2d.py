import torch
import triton
import triton.language as tl
from torch import Tensor
from .batch_norm import (
    _common_dtype_checks,
    _broadcast_shapes,
    _get_abcde,
    _unpack_train_x,
    _repack_output,
)
from .act_kernels import apply_relu
from .conv2d import (
    _get_mc,
    _config_strategy,
    _init_handles,
    _run_conv_fwd,
    _run_conv_bwd_d,
    _run_conv_bwd_w,
)
from .utils import (
    calculate_settings,
    get_config,
    volume,
)

def config_prune():
    return Config(
        num_stages=3,
        num_warps=8,
        min_num_stages=1,
        pre_hook=lambda args: [
            args["z_ptr"].zero_(),
            args["m_ptr"].zero_(),
        ],
    )

@triton.heuristics({
    'num_warps': lambda nargs: 4 if nargs['M'] <= 16 else 8,
})
@triton.jit
def dropout_kernel(
    x_ptr, y_ptr, z_ptr, m_ptr, drop_p, seed,
    C, H, W, CI, CO, KH, KW, NB,
    stride_x_c, stride_x_h, stride_x_w, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_C: tl.constexpr, GROUP_SIZE: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
    """
    Randomly zeroes elements in the input.

    Args:
        x_ptr: Pointer to the input to perform dropout on.
            The input must be of shape [C, H, W].
        y_ptr: Pointer to the output.
            The output must be of shape [C, H, W].
        z_ptr: Pointer to a counter of nonzero elements.
            The counter must be of shape [NB].
        m_ptr: Pointer to a mask of elements which were not dropped.
            The mask must be of shape [C, H, W].
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        C: Number of input feature maps.
        H: Height of input feature maps.
        W: Width of input feature maps.
        CI: Number of elements per feature map in the input.
            CI = H * W
        CO: Number of elements per feature map in the output.
            CO = H * W
        KH: Height of the convolutional kernel.
        KW: Width of the convolutional kernel.
        NB: Number of blocks in the output.
        stride_x_c: Stride necessary to jump one feature map in the input.
            stride_x_c = H * W
        stride_x_h: Stride necessary to jump one row in the input.
            stride_x_h = W
        stride_x_w: Stride necessary to jump one column in the input.
            stride_x_w = 1
        stride_y_c: Stride necessary to jump one feature map in the output.
            stride_y_c = H * W
        stride_y_h: Stride necessary to jump one row in the output.
            stride_y_h = W
        stride_y_w: Stride necessary to jump one column in the output.
            stride_y_w = 1
        BLOCK_C: Block size across feature maps.
            Must be divisible by GROUP_SIZE.
        GROUP_SIZE: Size of a group of feature maps processed together.
            GROUP_SIZE * (BLOCK_C // GROUP_SIZE) = BLOCK_C
        BLOCK_D: Block size across elements within a feature map.
            Must equal BLOCK_M * BLOCK_N.
        BLOCK_M: Block size across rows.
        BLOCK_N: Block size across columns.
    """
    # Set num_stages=1 to minimize NUM_REGS.
    # Reducing NUM_REGS reduces register spilling which can significantly
    # improve performance.
    pid = tl.program_id(0) * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    cols = pid % BLOCK_C
    c = cols // (BLOCK_D // GROUP_SIZE)
    d = cols % (BLOCK_D // GROUP_SIZE)
    m = d // BLOCK_N
    n = d % BLOCK_N

    # Move to the next feature map if out of bounds.
    if c >= C:
        return

    # Compute offsets for feature map access.
    x_offsets = c * CI + (m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_x_h + \
        (n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_x_w
    y_offsets = c * CI + (m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_y_h + \
        (n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_y_w
    x_mask = x_offsets < C * CI
    x_ptrs = x_ptr + x_offsets
    x_vals = tl.load(x_ptrs, mask=x_mask).to(tl.float32)
    y_ptrs = y_ptr + y_offsets
    m_ptrs = m_ptr + y_offsets

    use_fp16 = isinstance(x_vals, tl.half)

    # Convert fp32 to fp16 if necessary.
    if use_fp16:
        x_vals = x_vals.to(tl.half)

    random = tl.rand(seed, m * N * C + d * GROUP_SIZE + pid)
    x_vals = tl.where(random > drop_p, x_vals / (1 - drop_p), 0.0)

    # Write back to y and m if in-bounds.
    tl.store(y_ptrs, x_vals, mask=x_mask)
    tl.store(m_ptrs, random > drop_p, mask=x_mask)

    # Count the number of nonzero elements.
    nonzero = tl.sum(tl.where(random > drop_p, 1.0, 0.0))
    tl.atomic_add(z_ptr + pid, nonzero)

def dropout_bwd_kernel(
    x_ptr, out_grad_ptr, in_grad_ptr, m_ptr, drop_p, seed,
    C, H, W, CI, CO, KH, KW, NB,
    stride_x_c, stride_x_h, stride_x_w, stride_out_grad_c,
    stride_out_grad_h, stride_out_grad_w, stride_in_grad_c,
    stride_in_grad_h, stride_in_grad_w, BLOCK_C: tl.constexpr,
    GROUP_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ):
    """
    Calculates the input gradient of dropout.

    Args:
        x_ptr: Pointer to the input to compute the gradient of.
            The input must be of shape [C, H, W].
        out_grad_ptr: Pointer to the output's gradients.
            The output's gradients must be of shape [C, H, W].
        in_grad_ptr: Pointer to a container the input's gradients are written to.
            The container must be of shape [C, H, W].
        m_ptr: Pointer to a mask of elements which were not dropped in forward pass.
            The mask must be of shape [C, H, W].
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask in forward pass.
        C: Number of input feature maps.
        H: Height of input feature maps.
        W: Width of input feature maps.
        CI: Number of elements per feature map in the input.
            CI = H * W
        CO: Number of elements per feature map in the output.
            CO = H * W
        KH: Height of the convolutional kernel.
        KW: Width of the convolutional kernel.
        NB: Number of blocks in the output.
        stride_x_c: Stride necessary to jump one feature map in the input.
            stride_x_c = H * W
        stride_x_h: Stride necessary to jump one row in the input.
            stride_x_h = W
        stride_x_w: Stride necessary to jump one column in the input.
            stride_x_w = 1
        stride_out_grad_c: Stride necessary to jump one feature map
            in the output's gradients.
            stride_out_grad_c = H * W
        stride_out_grad_h: Stride necessary to jump one row in the output's
            gradients.
            stride_out_grad_h = W
        stride_out_grad_w: Stride necessary to jump one column in the output's
            gradients.
            stride_out_grad_w = 1
        stride_in_grad_c: Stride necessary to jump one feature map in the
            input's gradients.
            stride_in_grad_c = H * W
        stride_in_grad_h: Stride necessary to jump one row in the input's
            gradients.
            stride_in_grad_h = W
        stride_in_grad_w: Stride necessary to jump one column in the input's
            gradients.
            stride_in_grad_w = 1
        BLOCK_C: Block size across feature maps.
            Must be divisible by GROUP_SIZE.
        GROUP_SIZE: Size of a group of feature maps processed together.
            GROUP_SIZE * (BLOCK_C // GROUP_SIZE) = BLOCK_C
        BLOCK_D: Block size across elements within a feature map.
            Must equal BLOCK_M * BLOCK_N.
        BLOCK_M: Block size across rows.
        BLOCK_N: Block size across columns.
    """
    # Set num_stages=1 to minimize NUM_REGS.
    pid = tl.program_id(0) * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    cols = pid % BLOCK_C
    c = cols // (BLOCK_D // GROUP_SIZE)
    d = cols % (BLOCK_D // GROUP_SIZE)
