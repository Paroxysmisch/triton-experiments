import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X_p, Y_p, W_p, eps,
    BLOCK_SIZE: tl.constexpr,
    N: tl.constexpr,
    num_warps: tl.constexpr,
    program_id: tl.program_id(),
    warp_id: tl.warp_id(),
    lane_id: tl.lane_id(),
):
    # Load data from memory
    x = tl.load(X_p + program_id * BLOCK_SIZE + warp_id * num_warps + lane_id)
    w = tl.load(W_p + program_id * BLOCK_SIZE + warp_id * num_warps + lane_id)

    # Compute variance
    var = tl.sum(x * x, axis=0) / N

    # Compute rrms
    rrms = tl.rsqrt(var + eps)

    # Apply normalization and scale by weights
    y = (x * rrms).to(tl.float32) * w

    # Store result
    tl.store(Y_p + program_id * BLOCK_SIZE + warp_id * num_warps + lane_id, y)

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        # Allocate output tensor
        Y = torch.empty_like(x)

        # Compute grid size
        BLOCK_SIZE = 256
        grid_size = (x.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Call kernel
        rms_norm_kernel[grid_size, BLOCK_SIZE](
            x, Y, weight, eps,
            BLOCK_SIZE, x.shape[1], grid_size,
            program_id=0,
            warp_id=0,
            lane_id=0,
        )

        return Y

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
