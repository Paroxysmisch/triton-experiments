import triton
import triton.language as tl

@triton.autotune()
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    OUT_PTR,                            # output pointer
    INITIAL_STATE_PTR,                   # initial state pointer
    NT, K, V,                            # problem size
    STRIDE,                              # stride
    lambdaID: tl.constexpr,              # lambda id
    USE_INITIAL_STATE: tl.constexpr,     # flag to use initial state
    STORE_FINAL_STATE: tl.constexpr      # flag to store final state
):
    # load in initial state
    if USE_INITIAL_STATE:
        BLOCKIDX = tl.program_id(0)
        h = tl.load(INITIAL_STATE_PTR + BLOCKIDX * STRIDE)
    else:
        h = ...  # initialize h

    # iteration over time
    for n in range(NT):
        block_ptr = tl.make_block_ptr(OUT_PTR, [n, 0, 0])
        block_ptr = block_ptr.residing_ldg()

        # update h and v
        h = update_h(h, k, v, d)

        # compute and store new cumulative state
        if STORE_FINAL_STATE:
            tl.store(block_ptr, h)

        # compute cumulative sum
        b_h_cumsum = tl.dot(h, h, axis=1)

        # write back new 'v'
        v_new = b_h_cumsum[:, None] * v
        tl.store(block_ptr, v_new)

    # Return h if STORE_FINAL_STATE is True else return None
    if STORE_FINAL_STATE:
        return h
    else:
        return None

def chunk_delta_rule_fwd_fn(...):
    BT, BK, BV, NT, K, V, STRIDE, USE_INITIAL_STATE, STORE_FINAL_STATE = compute_sizes_and_strides_for_chunk_delta_rule_fwd_kernel(...)

    h = torch.empty(...)  # initialize h
    v_new = torch.empty(...)  # initialize v_new

    grid = lambda meta: (meta['batch'], )
    chunks_per_warp = (K // BT)

    chunk_delta_rule_fwd_kernel_h[grid](
        OUT_PTR=v_new.data_ptr(),
        INITIAL_STATE_PTR=h.data_ptr(),
        NT=NT, K=K, V=V,
        STRIDE=STRIDE,
        chunks_per_warp=chunks_per_warp,
        lambdaID=0,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        STORE_FINAL_STATE=STORE_FINAL_STATE,
    )

    return v_new
