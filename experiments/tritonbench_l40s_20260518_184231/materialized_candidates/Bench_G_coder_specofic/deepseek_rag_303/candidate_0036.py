import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2),
    ],
    key=["n_heads", "seqlen", "d_head"],
    reset_to_zero=["o_ptr"],  # important for autotuner, resets output to zeros
)
@triton.jit
def _score_kernel(
    Q_ptr, K_ptr, M_ptr, O_ptr,
    stride_h, stride_m, stride_n,
    n_heads, seqlen, d_head,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    # max number of <context, word> pairs allowed, affects sparsity code.
    SPARSIFY_CONTEXT_CACHE: tl.constexpr = 300,  # can't use parameter for autotune yet
    SHOULD_SPARSIFY_CONTEXT: tl.constexpr = False,  # can't use parameter for autotune yet
    # sliding window configurations
    WINDOW_M: tl.constexpr = 0, WINDOW_N: tl.constexpr = 0,
    BLOCK_M_REPEAT: tl.constexpr = 0, BLOCK_N_REPEAT: tl.constexpr = 0,
):
    """Kernel to compute attention scores.

    Q_ptr: the query matrix
    K_ptr: the key matrix
    M_ptr: the mask matrix
    O_ptr: the output matrix
    stride_h: Stride necessary to jump one head
    stride_m: Stride necessary to jump one row in Q or K
    stride_n: Stride necessary to jump one column in Q or K
    n_heads: Number of heads in the input
    seqlen: Number of tokens in the input
    d_head: Dimensionality of each head
    BLOCK_M: Block size to use for the M dimension
    BLOCK_N: Block size to use for the N dimension
    SPARSIFY_CONTEXT_CACHE: max allowed sparsity cache block size
    SHOULD_SPARSIFY_CONTEXT: boolean indicating whether to sparsify context
    WINDOW_M: size of window on m
    WINDOW_N: size of window on n
    BLOCK_M_REPEAT: device-side over-allocation multiplier for block_m
    BLOCK_N_REPEAT: device-side over-allocation multiplier for block_n
    """
    # We use BLOCK_M_REPEAT and BLOCK_N_REPEAT to allow for extra
    # parallelization on the BLOCK_(M,N) (where BLOCK_(M,N) is a multiple of device-side block sizes),
    # used for extra load balancing. Usage is extend to non-integer cases,
    # since parameter can't be of tl.constexpr integer type.
    BLOCK_M_REPEAT = tl.constexpr(int(BLOCK_M_REPEAT))
    BLOCK_N_REPEAT = tl.constexpr(int(BLOCK_N_REPEAT))

    # get start row and column
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)

    # slide over K (m, n)
    # we will use `launch_grid` as parameters for `_score_kernel`
    mask_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # boundry check on m & n
    mask_m_bound = mask_m < n_heads

    Q_ptr_chunk = Q_ptr + (pid_h * stride_h + mask_m * stride_m)
    Q_ptr_chunk = tl.make_block_ptr(Q_ptr_chunk, shape=(d_head, seqlen), 
                                    strides=(1, stride_m), offsets=(0, 0), 
                                    block_shape=(BLOCK_M_REPEAT * BLOCK_M, BLOCK_N), 
                                    order=(1, 0))
    # nead to do this because our BLOCK_M might not divide evenly
    # into n_heads, using tl.arange with BLOCK_M for getting correct part
    # of the `Q_ptr_h` would trigger assert if BLOCK_M>n_heads, so having
    # another condition is needed.
    Q_ptr_chunk = tl.subblock_ptr(Q_ptr_chunk, order=(1, 0), key=(0, 0), 
                                  block_shape=(BLOCK_M_REPEAT * BLOCK_M, BLOCK_N))

    for block_n in range(0, tl.cdiv(seqlen, BLOCK_N_REPEAT * BLOCK_N)):
        # only slide over 128 context at a time, but handle window over 256
        moving_block_n = block_n * BLOCK_N
        moving_block_n = tl.multiple_of(moving_block_n, BLOCK_N)
        mask_n = ((moving_block_n + tl.arange(0, BLOCK_N_REPEAT * BLOCK_N)) 
                  % seqlen)
        mask_n_bound = mask_n < seqlen

        K_ptr_chunk = K_ptr + (pid_h * stride_h + mask_n * stride_n)
        K_ptr_chunk = tl.make_block_ptr(K_ptr_chunk, shape=(seqlen, d_head),
                                        strides=(stride_n, 1), offsets=(0, 0),
                                        block_shape=(BLOCK_N, BLOCK_M_REPEAT * BLOCK_M),
                                        order=(0, 1))
        K_ptr_chunk = tl.subblock_ptr(K_ptr_chunk, order=(1, 0), key=(0, 0),
                                      block_shape=(BLOCK_N, BLOCK_M_REPEAT * BLOCK_M))

        qk = tl.zeros([BLOCK_M_REPEAT * BLOCK_M, BLOCK_N], dtype=tl.float32)
        count_mask = tl.zeros([BLOCK_M_REPEAT * BLOCK_M, BLOCK_N], dtype=tl.int32)
        # TODO: we could get arround non-continuous block
        # on the bottom right when keyleaning is enable
        # could try to load Q, K blockwise instead of all at once
        Q_tile = tl.load(Q_ptr_chunk, boundary_check=(0, 1), 
                         padding_option="zero").to(Q_ptr.dtype.element_ty)

        sm_scale = d_head ** -0.5

        # move over k,

        K_tile = tl.load(K_ptr_chunk, boundary_check=(0, 1), 
                         padding_option="zero", advance_to_padding=True)

        K_tile = K_tile.to(Q_ptr.dtype.element_ty)
        qk += tl.dot(Q_tile, K_tile)
        count_mask += tl.where(
            tl.arange(0, BLOCK_M_REPEAT * BLOCK_M)[:, None] < n_heads 
            and tl.arange(0, BLOCK_N)[None, :] < seqlen,
