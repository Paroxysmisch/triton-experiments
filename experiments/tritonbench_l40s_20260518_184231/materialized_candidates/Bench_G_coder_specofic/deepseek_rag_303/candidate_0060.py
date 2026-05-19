import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 16}, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 32}, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 64}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 64}, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE_H': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE_H': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 256}, num_warps=16),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _triton_rope(
    q_ptr, k_ptr,
    cos, sin,
    chunk_size,
    hdim,
    dq_ptr, dk_ptr,
    BLOCK_SIZE_H: tl.constexpr, BACKWARD_PASS: tl.constexpr,
):
    """
    Apply RPE to q and k. Triton-annotated function
    """
    # Unique program ID:
    pid = tl.program_id(axis=0)
    # get current batch:
    cur_batch = pid
    # get current head:
    cur_head = tl.program_id(axis=1)
    # Trigonometric identities:
    half_hdim = hdim // 2
    # Offsets for tl.load and tl.store:
    offs_q1 = cur_batch * chunk_size               + cur_head * hdim + tl.arange(0, BLOCK_SIZE_H)
    offs_q2 = cur_batch * chunk_size               + cur_head * hdim + half_hdim + tl.arange(0, BLOCK_SIZE_H)
    offs_cos = tl.arange(0, BLOCK_SIZE_H // 2)
    offs_sin = tl.arange(0, BLOCK_SIZE_H // 2)
    # Load from q and k:
    sq1 = tl.load(q_ptr + offs_q1)
    sq2 = tl.load(q_ptr + offs_q2)
    dk1 = tl.load(k_ptr + offs_q1)
    dk2 = tl.load(k_ptr + offs_q2)
    # Phase shift:
    k_phase = tl.full([BLOCK_SIZE_H // 2, ], (half_hdim * 2) * cur_head, tl.int32)
    cos_ptr = cos + k_phase + tl.arange(0, BLOCK_SIZE_H // 2)
    sin_ptr = sin + k_phase + tl.arange(0, BLOCK_SIZE_H // 2)
    cs = tl.load(cos_ptr, mask=offs_cos < half_hdim, other=0)
    sn = tl.load(sin_ptr, mask=offs_sin < half_hdim, other=0)
    # Apply rotation:
    if BACKWARD_PASS:
        csn = -sn
    rq1 = sq1 * cs - sq2 * csn
    rq2 = sq2 * cs + sq1 * csn
    rk1 = dk1 * cs - dk2 * csn
    rk2 = dk2 * cs + dk1 * csn
    # Store rotated:
    offs_dq1 = cur_batch * chunk_size               + cur_head * hdim + tl.arange(0, BLOCK_SIZE_H)
    offs_dq2 = cur_batch * chunk_size               + cur_head * hdim + half_hdim + tl.arange(0, BLOCK_SIZE_H)
    offs_dk1 = cur_batch * chunk_size               + cur_head * hdim + tl.arange(0, BLOCK_SIZE_H)
    offs_dk2 = cur_batch * chunk_size               + cur_head * hdim + half_hdim + tl.arange(0, BLOCK_SIZE_H)
    tl.store(dq_ptr + offs_dq1, rq1, mask=offs_q1 < chunk_size)
    tl.store(dq_ptr + offs_dq2, rq2, mask=offs_q2 < chunk_size)
    tl.store(dk_ptr + offs_dk1, rk1, mask=offs_k1 < chunk_size)
    tl.store(dk_ptr + offs_dk2, rk2, mask=offs_k2 < chunk_size)

@torch.no_grad()
def rope_backward(q, k, dq, dk, cos, sin):
    """
    Wrapper function for calling RPE triton kernel
    """
    batch, chunk, head, hdim = q.shape
    # Ensure dimensions are power of two:
    if hdim < 0:
        hdim = 2048
    else:
        hdim = 1
        while hdim < sig.shape[-1]:
            hdim *= 2
    assert all((q.shape == dq.shape, q.shape == dk.shape, k.shape == dk.shape))
    assert cos.shape[0] >= hdim // 2
    assert sin.shape[0] >= hdim // 2
    NUM_WARPS = 8
    BLOCK_SIZE_H = hdim
    _triton_rope[(batch * head, head), ](
        q_ptr=q, k_ptr=k,
        cos=cos, sin=sin,
        chunk_size=chunk,
        hdim=hdim,
        dq_ptr=dq, dk_ptr=dk,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BACKWARD
