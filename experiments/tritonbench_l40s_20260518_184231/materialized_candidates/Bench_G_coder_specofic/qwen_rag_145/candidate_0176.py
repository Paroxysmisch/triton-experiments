import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    # input tensors
    logics,
    # output tensors
    output,
    # input parameters
    BLOCK_DMODEL: tl.constexpr,
    # tensor strides
    stride_logic_h,
    stride_logic_bs,
    seq_len,
    batch_id,
    head_id,
):
    # get the current batch and head
    batch = tl.program_id(0)
    head = tl.program_id(1)
    
    # load the sequence length and input parameters
    sequence_length = tl.load(seq_len + batch)
    input_pointer = logics + head * stride_logic_h + batch * stride_logic_bs
    
    # prepare the accumulator and maximum value
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    e_max = float("-inf")
    
    # apply the softmax
    for i in range(sequence_length):
        input_val = tl.load(input_pointer + i)
        n_e_max = tl.maximum(input_val, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(input_val - n_e_max)
        e_sum = p
        acc = acc * old_scale + p * input_val
        e_max = n_e_max
    
    # store the result
    output_pointer = output + head * stride_logic_h + batch * stride_logic_bs
    tl.store(output_pointer, acc / e_sum)

@torch.no_grad()
def token_softmax_fwd(logics, output, seq_len):
    # get the shape of the input data
    BLOCK = 64
    batch, head = seq_len.shape[0], logics.shape[0]
    
    # launch the kernel
    grid = (batch, head)
    _fwd_kernel_token_softmax[grid](
        logics, output,
        BLOCK_DMODEL=logics.shape[-1],
        stride_logic_h=logics.stride(0),
        stride_logic_bs=logics.stride(1),
        seq_len=seq_len,
        batch_id=torch.arange(0, batch),
        head_id=torch.arange(0, head),
        num_warps=1,
        num_stages=1
    )
