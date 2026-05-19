import torch
import triton
import triton.language as tl

@torch.no_grad()
def token_att_fwd2(
    Prob,
    V,
    Req_to_tokens,
    batch_ids,
    head_ids,
    seq_lengths,
    head_ids_start,
    head_ids_end,
    head_group_num,
    BLOCK_N=128,
    BLOCK_K=128,
    BLOCK_M=128,
    BLOCK=128,
    HEAD_GROUP_NUM=1,
):
    # Initialize parameters
    grid_size = (len(batch_ids) * len(head_ids) * (len(seq_lengths) // BLOCK_N) * (len(head_group_num) // BLOCK_K),)
    block_size = (BLOCK, 1, 1)
    
    # Compute kv_group_num
    kv_group_num = head_group_num
    
    # Call the Triton kernel
    _fwd_kernel_token_att2[grid_size, block_size](
        Prob,
        V,
        Out,
        Req_to_tokens,
        batch_ids,
        head_ids,
        seq_lengths,
        head_ids_start,
        head_ids_end,
        head_group_num,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_M=BLOCK_M,
        BLOCK=BLOCK,
        HEAD_GROUP_NUM=HEAD_GROUP_NUM,
    )
