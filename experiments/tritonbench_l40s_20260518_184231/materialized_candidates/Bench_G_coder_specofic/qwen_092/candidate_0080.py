triton
import triton
import triton.language as tl

# Define the block sizes
BLOCK_DMODEL = 64
BLOCK_N = 128

# Triton kernel function
@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Req_to_tokens, Out,
    B_req_idx, B_Start_Loc, B_Seqlen,
    BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    stride_P: tl.constexpr, stride_V: tl.constexpr, stride_RT: tl.constexpr, stride_O: tl.constexpr,
    stride_B_req_idx: tl.constexpr, stride_B_Start_Loc: tl.constexpr, stride_B_Seqlen: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # Calculate offsets
    p_off = cur_batch * stride_B_req_idx + cur_head * stride_P
    v_off = cur_batch * stride_B_req_idx + cur_head * stride_V
    rt_off = cur_batch * stride_B_req_idx + cur_head * stride_RT
    o_off = cur_batch * stride_B_req_idx + cur_head * stride_O

    # Initialize accumulator
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)

    # Loop over tokens in the block
    for n in range(0, BLOCK_N, BLOCK_DMODEL):
        # Load probabilities and values
        prob = tl.load(Prob + p_off + n, mask=n < BLOCK_N, other=0.0)
        v = tl.load(V + v_off + n, mask=n < BLOCK_N, other=0.0)

        # Load token indices
        token_indices = tl.load(Req_to_tokens + rt_off + n, mask=n < BLOCK_N, other=0)

        # Compute weighted sum
        for i in range(BLOCK_DMODEL):
            acc[i] += prob[i] * v[i]

    # Store the result back to the output tensor
    tl.store(Out + o_off, acc)

# Wrapper function
@triton.jit
def token_att_fwd2(
    Prob, V, Req_to_tokens, Out,
    B_req_idx, B_Start_Loc, B_Seqlen,
    kv_group_num: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Set grid size
    grid = (kv_group_num, kv_group_num)

    # Launch the kernel
    _fwd_kernel_token_att2[grid](
        Prob, V, Req_to_tokens, Out,
        B_req_idx, B_Start_Loc, B_Seqlen,
        BLOCK_DMODEL, BLOCK_N,
        Prob.stride(0), V.stride(0), Req_to_tokens.stride(0), Out.stride(0),
        B_req_idx.stride(0), B_Start_Loc.stride(0), B_Seqlen.stride(0),
        num_warps, num_stages
    )

# Example usage
# Assuming Prob, V, Req_to_tokens, Out, B_req_idx, B_Start_Loc, B_Seqlen are already defined
# token_att_fwd2(Prob, V, Req_to_tokens, Out, B_req_idx, B_Start_Loc, B_Seqlen, kv_group_num, num_warps, num_stages)
