import triton
import triton.language as tl
import torch


@triton.jit
def _fused_instance_norm_selu_conv2d_kernel(
    INPUT_PTR,           # [N, CIN, H, W]
    WEIGHT_PTR,          # [COUT, CIN/groups, KH, KW]
    BIAS_PTR,            # [COUT] or None
    RUNNING_MEAN_PTR,    # [N * COUT] or None if track_running_stats=False
    RUNNING_VAR_PTR,     # [N * COUT] or None if track_running_stats=False
    GAMMA_PTR,           # [COUT] or None if affine=False
    BETA_PTR,            # [COUT] or None if affine=False
    OUTPUT_PTR,          # [N, COUT, H_out, W_out]
    # Shapes / Parameters
    N, CIN, H, W,
    COUT, KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    GROUPS,
    EPS, MOMENTUM,
    AFFINE, TRACK_STATS,
    # Block sizes for parallelism
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    """
    A single Triton kernel that:
      1) Performs 2D convolution on one tile of output.
      2) Applies SELU activation.
      3) Computes instance statistics (mean/var) per instance/channel tile.
      4) Applies instance normalization (optionally with affine transform).
    """

    # Program IDs to identify which tile we are computing
    # Each program handles a tile of size [BLOCK_H, BLOCK_W] in H_out x W_out
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bco = tl.program_id(2)  # combined dimension for batch + out_channels

    # Decompose the pid_bco into (n, co) index
    # We will map this program id into a unique (batch, channel) pair
    # so that each kernel instance handles one tile for a specific (n, co).
    n = pid_bco // COUT
    co = pid_bco % COUT

    # Output spatial start
    h_out_start = pid_h * BLOCK
