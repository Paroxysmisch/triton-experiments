import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def attention_fwd_kernel(
    q, k, v, h, o, log_sum_weights, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    s_h_h, s_h_t, s_h_d, 
    T, BT, 
    H, 
    H_DIM, 
    num_warps, 
    VAR_LEN: tl.constexpr, 
    STAGE: tl.constexpr, 
    IFCOND: tl.constexpr, 
    STORE: tl.constexpr
    ):
    # block id
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    # get the number of blocks
    if VAR_LEN:
        b_num = tl.num_programs(0) - 1
    else:
        b_num = tl.num_programs(0)
    # get the slice stride
    bt = BT*block_id
    bt_end = bt + BT if VAR_LEN and block_id < b_num - 1 else T
    # initialize b_h to zero
    b_h = tl.zeros([1, 1, H_DIM], dtype=tl.float32)
    for start_t in range(bt, bt_end, BT):
        start_t = tl.multiple_of(start_t, BT)
        # pointers
        p_q = q + start_t * s_qk_h * s_qk_t + head_id * s_qk_h + tl.arange(0, BT)[:, None] \
            * s_qk_t + tl.arange(0, H_DIM)[None, :]  
        p_k = k + start_t * s_qk_h * s_qk_t + (0 * s_qk_h) + tl.arange(0, H_DIM)[None, :] \
            * s_qk_t + tl.arange(0, BT)[:, None]  
        if IFCOND:
            p_h = h + head_id * s_h_h + start_t * s_h_d  
        p_v = v + start_t * s_vo_h * s_vo_t + head_id * s_vo_h + tl.arange(0, BT)[:, None] \
            * s_vo_t + tl.arange(0, H_DIM)[None, :]  
        if STORE:
            p_o = o + start_t * s_vo_h * s_vo_t + head_id * s_vo_h + tl.arange(0, BT)[:, None] \
                * s_vo_t + tl.arange(0, H_DIM)[None, :]
        # load q, k, v
        b_q = tl.load(p_q)
        b_k = tl.load(p_k)
        b_v = tl.load(p_v)
        # k, v are 3D. [head, BT, H_DIM]
        b_s = tl.dot(b_q, b_k)
        b_s = b_s * 1.44269504
        if IFCOND:
            # conditional update for h
            b_h = b_h * 0.0
            b_o = tl.dot(b_s.to(b_h.dtype), b_h, out_dtype=b_h.dtype)
            b_s = b_s + tl.load(p_h).to(b_h.dtype)
            b_h = tl.dot(b_v, (b_s + 1.0)[:, None])
            if STORE:
                tl.store(p_h, b_s.to(p_h.dtype.element_ty))
        else:
            # standard update for h
            b_h = b_h + tl.dot(b_k.to(b_h.dtype), b_v.to(b_h.dtype), out_dtype=b_h.dtype)
        # update o
        b_o = tl.dot(b_s.to(b_h.dtype), b_h, out_dtype=b_h.dtype)
        if STORE:
            tl.store(p_o, b_o.to(p_o.dtype.element_ty))
    # record log weights
    if log_sum_weights is not None:
        p_log_w = log_sum_weights + head_id * BT + block_id * 1024
        tl.store(p_log_w)   # placeholder size: 1, dtype: float32


class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, h, is_first_stage, is_last_stage, ABSOLUTE_TOLERANCE, STORE):
        # check constraints
        assert q.shape == k.shape
        assert q.shape == v.shape
        BT = get_block_size(q)
        if is_first_stage:
            H = q.shape[1]//BT
        else:
            H = 1
        T, BT = q.shape[2], q.shape[2]
        assert q.shape[1] == BT * H
        # allocate workspace for o
        BT_M = triton.next_power_of_2(BT)
        # allocate workspace
        o = torch.empty_like(q, dtype=torch.float16)
        log_sum_weights = torch.empty(
            [q.shape[0] * q.shape[1] * BT, q.shape[2]], dtype=torch.float32, 
            device=q.device) if BT < 1024 else None
        # kernel
        num_warps = 4 if version.parse(triton.__version__) >= version.parse("2.1.0") else 8
        if not is_last_stage:
            # re-calculate
            ABSOLUTE_TOLERANCE = 1
        # launch kernel
        attention_fwd_kernel[(q.shape[2]//BT_M + 1, q.shape[0]*q.shape[1])]\
            (
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), 
            h.to(torch.float16), o.to(torch.float16), log_sum_weights, 
            q.stride(0), q.stride(1), q.stride(2), 
            k.stride(0), k.stride(1), k.stride(2), 
            v.stride(0), v.stride(1), v.stride(2), 
            h.stride(0), h.stride(1), h.stride(2), 
            T, BT, H, q.shape[-1], 
            num_warps=num_warps, 
            VAR_LEN=True, 
            STAGE=q.shape[2]//BT, 
            IFCOND=is_first_stage, 
            ABSOLUTE_TOLERANCE=ABSOLUTE_TOLERANCE, 
            STORE=STORE
            )
        return q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), o.to(torch.float16),\
               h.to(torch.float16), log_sum_weights
