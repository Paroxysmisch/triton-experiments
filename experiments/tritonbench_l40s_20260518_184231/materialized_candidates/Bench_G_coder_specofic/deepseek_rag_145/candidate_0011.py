import triton.language as tl

@triton.jit
def rms_matmul_rbe(
    x_ptr: tl.constexpr,
    w_ptr: tl.constexpr,
    rms_w_ptr: tl.constexpr,
    output_ptr: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    THETA: tl.constexpr,
):
    # compute the linear id in the input
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.program_id(2)

    # calculate base pointers
    x_base_ptr = x_ptr + m * HEAD_SIZE * BLOCK_SIZE_K
    w_base_ptr = w_ptr + k * HEAD_SIZE * BLOCK_SIZE_N
    rms_w_base_ptr = rms_w_ptr + k * HEAD_SIZE
    output_base_ptr = output_ptr + m * HEAD_SIZE * BLOCK_SIZE_N

    # load x, w, and rms_w for the head
    x = tl.load(x_base_ptr + k * BLOCK_SIZE_K)
    w = tl.load(w_base_ptr + n * BLOCK_SIZE_N)
    rms_w = tl.load(rms_w_base_ptr + n)

    # apply RMS normalization and compute dot product
    x_norm = x / tl.sqrt(rms_w)
    dot_product = tl.sum(x_norm * w)

    # apply rotary embeddings if specified
    if THETA:
        rotary_embedding = tl.cos(THETA * k) * x + tl.sin(THETA * k) * w
        dot_product += tl.sum(rotary_embedding * w)

    # store the result
    tl.store(output_base_ptr + n * BLOCK_SIZE_N, dot_product)
