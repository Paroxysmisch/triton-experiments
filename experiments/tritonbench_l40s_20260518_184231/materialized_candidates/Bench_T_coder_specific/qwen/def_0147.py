import triton
import triton.language as tl

@triton.jit
def normalize_and_compute_distances(
    x1_ptr: tl.tensor,
    x2_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n: int,
    d: int,
    p_norm: float,
    eps_norm: float,
    eps_distance: float,
    block_size: int = 32,
):
    pid = tl.program_id(axis=0)
    x1_idx = pid * block_size + tl.arange(0, block_size)
    x2_idx = pid * block_size + tl.arange(0, block_size)

    # Load data from global memory
    x1 = tl.load(x1_ptr + x1_idx[:, None] * d)
    x2 = tl.load(x2_ptr + x2_idx[:, None] * d)

    # Normalize along dimension 1
    norms_x1 = tl.sum(tl.square(x1), axis=1, keepdims=True)
    norms_x1 = tl.maximum(norms_x1, eps_norm)
    x1_normalized = x1 / tl.sqrt(norms_x1)

    norms_x2 = tl.sum(tl.square(x2), axis=1, keepdims=True)
    norms_x2 = tl.maximum(norms_x2, eps_norm)
    x2_normalized = x2 / tl.sqrt(norms_x2)

    # Compute pairwise distances
    diff = x1_normalized - x2_normalized
    dists = tl.sum(tl.square(diff), axis=1)

    # Store results in global memory
    tl.store(out_ptr + x1_idx[:, None], dists)
