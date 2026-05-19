import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    req_to_tokens,
    stride_opbs,
    stride_oh,
    stride_vbs,
    stride_vh,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)

    block_start_loc = BLOCK_N * cur_token

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_token = block_start_loc + offs_n

    req_indices = tl.load(req_to_tokens + offs_token)
    p_ptrs = Prob + (req_indices * stride_opbs + cur_head * stride_oh + offs_d[None, :])
    v_ptrs = V + (req_indices * stride_vbs + cur_head * stride_vh + offs_d[None, :])

    p_value = tl.load(p_ptrs, mask=offs_n < BLOCK_N, other=0.0)
    v_value = tl.load(v_ptrs, mask=offs_n < BLOCK_N, other=0.0)

    acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    acc += tl.dot(p_value, v_value)

    tl.store(Out + offs_token * stride_oh + cur_head * stride_oh, acc, mask=offs_n < BLOCK_N)


@torch.no_grad()
def token_att_fwd2(Prob, V, Out, req_to_tokens, max_input_len):
    if torch.cuda.get_device_capability()[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq = Prob.shape[-1]
    assert Lq == V.shape[-1]
    assert Lq in {16, 32, 64, 128, 256}

    batch, head = Prob.shape[0], Prob.shape[1]
    kv_group_num = Prob.shape[1] // V.shape[1]

    grid = (max_input_len, head)
    num_warps = 4 if Lq <= 64 else 8

    _fwd_kernel_token_att2[grid](
        Prob,
        V,
        Out,
        req_to_tokens,
        Prob.stride(0),
        Out.stride(1),
        V.stride(0),
        V.stride(1),
        BLOCK_N=BLOCK,
        BLOCK_D=Lq,
        num_warps=num_warps,
        num_stages=1,
    )
