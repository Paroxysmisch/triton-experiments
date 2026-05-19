import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    BLOCK_M,
    BLOCK_N,
    head_dim,
    seq_len,
    q_scale,
    k_scale,
    BLOCK_SIZE=1024,
):
    pid = tl.program_id(axis=0)
    num_warps = BLOCK_SIZE // 32
    num_programs = tl.program_count(axis=0)

    # ... implementation goes here ...

def forward(q, k, v, q_scale, k_scale):
    # ... setup goes here ...

    _attn_fwd[(num_blocks, 1, 1)](
        q_ptr,
        k_ptr,
 v_ptr,
        o_ptr,
        BLOCK_M,
        BLOCK_N,
        head_dim,
        seq_len,
        q_scale,
        k_scale,
    )

    return o
