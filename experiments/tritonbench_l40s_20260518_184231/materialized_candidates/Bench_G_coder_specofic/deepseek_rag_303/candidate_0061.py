import torch
import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    Input, 
    Weights, 
    lora_indices, 
    Output, 
    M, 
    N, 
    K, 
    slice_size, 
    lora_iters,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BATCH_SIZE: tl.constexpr, 
    D_HEAD: tl.constexpr, 
    EVEN_K: tl.constexpr,
    CONV_Y: tl.constexpr,
    PRE_ADD: tl.constexpr,
    OUT_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cur_batch = pid // slice_size
    cur_lora = pid % slice_size

    if cur_batch >= BATCH_SIZE or cur_lora >= lora_iters:
        return

    lo_idx = cur_batch * slice_size * D_HEAD * D_HEAD + cur_lora
    hi_idx = lo_idx + D_HEAD * D_HEAD

    WeightBlockPtr = tl.make_block_ptr(
        base=Weights + (lo_idx // D_HEAD),
        shape=(K, D_HEAD),
        strides=(1, D_HEAD),
        offsets=(0, (cur_lora % D_HEAD)*D_HEAD),
        block_shape=(1, D_HEAD),
        order=(0, 1),
    )

    InputBatchStartPtr = tl.make_block_ptr(
        base=Input + (cur_batch * K * M),
        shape=(K, M),
        strides=(1, M),
        offsets=(0, 0),
        block_shape=(1, M),
        order=(0, 1),
    )

    ResultMatPtr = tl.make_block_ptr(
        base=Output + (cur_batch * CONV_Y * lora_iters * D_HEAD),
        shape=(CONV_Y, lora_iters * D_HEAD),
        strides=(CONV_Y, D_HEAD),
        offsets=(0, cur_lora * D_HEAD),
        block_shape=(1, D_HEAD),
        order=(0, 1),
    )

    lo_j = cur_lora * D_HEAD * D_HEAD
    Hi_j = lo_j + D_HEAD * D_HEAD

    rvec = tl.arange(0, D_HEAD)
    cvec = tl.arange(0, D_HEAD)

    m_i = cur_lora % 1
    lora_idx_ptr = lora_indices + (m_i)

    fm_i = tl.load(lora_idx_ptr)

    _out_block_ptr = tl.make_block_ptr(
        base=Output + fm_i,
        shape=(D_HEAD, D_HEAD),
        strides=(N, 1),
        offsets=(rvec, cvec),
        block_shape=(D_HEAD, 1),
        order=(1, 0),
    )

    w = tl.load(WeightBlockPtr)
    _out = tl.zeros((D_HEAD,), dtype=tl.float32)

    m = 0

    while m < CONV_Y:

        input_ptrs = tl.advance(InputBatchStartPtr, m*M)
        block_ptrs = tl.advance(_out_block_ptr, m*D_HEAD)

        v = tl.load(input_ptrs, boundary_check=(0, ), padding_option='zero')
        _out += tl.dot(w, v, allow_tf32=True)

        m += 1

    if CONV_Y > 1:

        avg_val = tl.sum(_out, axis=0) / CONV_Y

        tl.store(block_ptrs, avg_val, boundary_check=(0,))

    else:

        tl.store(block_ptrs, _out, boundary_check=(0,))

    return

def _sgmv_expand_slice(

    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    lora_indices_4_b: torch.LongTensor,
    N: int,
    K: int,
    batch_size: int,
    lora_iters: int,
    d_head: int,
    slice_size_4_b: int,
    conv_y: int,
):

    # Preparation
    BLOCK_M = 1
    BLOCK_N = d_head
    pre_add = False

    output_buffer = torch.zeros(
        (batch_size, conv_y, lora_iters, d_head, d_head),
        dtype=torch.float32,
        device=input.device,
    )

    _sgmv_expand_slice_kernel[(batch_size * slice_size_4_b,)](
        input,
        weight,
        lora_indices_4_b,
        output_buffer,
        M=N,
        N=N,
        K=K,
        slice_size=slice_size_4_b,
        lora_iters=lora_iters,
        BLOCK_M=BLOCK_M,
        BATCH_SIZE=batch_size,
        D_HEAD=d_head,
        EVEN_K=K,
        PRE_ADD=pre_add,
        CONV_Y=conv_y,
        OUT_TYPE=output_buffer.dtype,
    )

    return output_buffer
