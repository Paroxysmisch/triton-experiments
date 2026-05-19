import triton
import triton.language as tl
import torch

@triton.jit
def q_kernel_per_block_int8(q, q_int8, q_scale, BLKQ, BLKQ, BLKHWM,
                            MASK, sm_scale, Q_STEPSSET,
                            stride_q_batch, stride_q_heads, stride_q_head_dim,
                            stride_q8_batch, stride_q8_heads, stride_q8_head_dim,
                            stride_scale_batch, stride_scale_heads, stride_scale_dim: tl.constexpr,
                            BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_HWM: tl.constexpr,
                            BLOCK_SIZE_HEAD: tl.constexpr, BLOCK_SIZE_HEAD_DIV_NUM_SMS: tl.constexpr,
                            NUM_SMS: tl.constexpr):
    # Triton kernel settings for converting query matrix to int8
    sm_idx = tl.program_id(2)
    q8_batch_offs = tl.program_id(0) * stride_q8_batch
    q8_head_offs = tl.program_id(1) * stride_q8_heads
    Q_STEPS = tl.cdiv(BLKQ, BLOCK_SIZE_Q)
    Q_HWM = BLKQ + BLOCK_SIZE_HWM
    BLKHW = BLKQ
    BLK_HEAD_DIM = BLKHWM
    BLK_HEAD_DIM_NUM_SMS = BLKHWM

    ROW_BLOCKS = tl.cdiv(BLK_HEAD_DIM, BLOCK_SIZE_HEAD)

    if sm_idx < NUM_SMS:
        offs_head = sm_idx * BLOCK_SIZE_HEAD_DIV_NUM_SMS
        offs_q = q8_batch_offs + offs_head * stride_q8_heads
        offs_q8 = tl.arange(0, BLOCK_SIZE_Q)
        offs_q_head = q8_head_offs + sm_idx * stride_q8_heads

        VALD_QSTEP = tl.full((BLOCK_SIZE_Q,), -1, dtype=tl.int8)
        VALD_QSTEP_MASK = tl.full((BLOCK_SIZE_Q,), 1, dtype=tl.int8)
        VALD_QSTEP_MASK = VALD_QSTEP_MASK == 1

        for qstep in range(0, Q_STEPS):
            offs_q += BLOCK_SIZE_Q
            offs_q8 += BLOCK_SIZE_Q
            q_ptrs = q + offs_q
            q8_ptrs = q_int8 + offs_q8
            q_head_ptrs = q_int8 + offs_q_head
            transposed_q8_ptrs = q_int8 + offs_head * stride_q8_heads + offs_q8

            load_ptrs = q_ptrs[:, None] + (offs_q8[None, :] * stride_q_head_dim)
            load_mask = (offs_q8[None, :] < BLK_HEAD_DIM)
            loaded_q = tl.load(load_ptrs, mask=load_mask, other=0.0)

            loaded_q = loaded_q.to(loaded_q.dtype) * sm_scale * \
                       loaded_q * sm_scale
            loaded_q = tl.where(loaded_q > 127, 127, loaded_q)
            loaded_q = tl.where(loaded_q < -128, -128, loaded_q)
            loaded_q = loaded_q.to(tl.int8)()

            loaded_q = tl.trans(tl.reshape(
                loaded_q, (ROW_BLOCKS, BLOCK_SIZE_Q))).to(loaded_q.dtype)

            tl.store(q8_ptrs, loaded_q.to(q8_ptrs.dtype.element_ty),
                     mask=offs_q8 < BLK_HEAD_DIM_NUM_SMS)
            tl.store(transposed_q8_ptrs, loaded_q,
                     mask=offs_q8 < BLK_HEAD_DIM_NUM_SMS)
            tl.store(q_head_ptrs, sm_idx)

            if qstep > 0:
                offs_q -= (qstep * BLOCK_SIZE_Q + BLOCK_SIZE_Q)
                offs_q8 -= (qstep * BLOCK_SIZE_Q + BLOCK_SIZE_Q)

        offs_q += Q_HWM
        offs_q8 += Q_HWM
        q_head_ptrs += Q_HWM

        q_extra_stores = tl.arange(Q_HWM, BLK_HEAD_DIV_NUM_SMS)

        # clear extra stores
        zero_q8_store = tl.zeros((1, ), dtype=tl.int8)
        tl.store(q8_ptrs + q_extra_stores, zero_q8_store, mask=(
            q_extra_stores < BLK_HEAD_DIV_NUM_SMS))

        HEAD_Q_OFFSET_DTYPE = tl.pointer_type(tl.int8)
        VALD_QSTEP_PTR = tl.make_block_ptr(
            VALD_QSTEP, (BLOCK_SIZE_Q, ),
            (1, ),
            (0, ),
            (BLOCK_SIZE_Q, ),
            (0, )
        )
        tl.store(VALD_QSTEP_PTR, VALD_QSTEP_MASK)

        head_q_offs = tl.arange(0, BLOCK_SIZE_Q) + \
            BLK_HEAD_DIV_NUM_SMS + BLOCK_SIZE_HEAD
        tl.store(q_head_ptrs + q_extra_stores, head_q_offs, mask=q_extra_stores < BLK_HEAD_DIM_NUM_SMS)
        q_extra_stores += Q_HWM
        tl.store(q_head_ptrs + q_extra_stores, BLK_HEAD_DIV_NUM_SMS, mask=q_extra_stores < BLK_HEAD_DIM_NUM_SMS)

        scale_offs = q8_batch_offs + q8_head_offs * stride_scale_heads + sm_idx * stride_scale_dim
        transposed_scale_offs = q8_head_offs * stride_scale_heads + q8_batch_offs + sm_idx * stride_scale_dim

        new_scale_2 = loaded_q * loaded_q
        tl.store(q_scale + scale_offs, new_scale_2)
        tl.store(q_scale + transposed_scale_offs, new_scale_2, mask=offs_q8 < BLK_HEAD_DIM_NUM_SMS)


@triton.jit
def k_kernel_per_block_int8(k, k_int8, k_scale, BLKQ, BLKK, BLKHWM,
                            MASK, sm_scale, K_STEPSSET,
                            stride_k_batch, stride_k_heads, stride_k_head_dim,
                            stride_k8_batch, stride_k8_heads, stride_k8_head_dim,
                            stride_scale_batch, stride_scale_heads, stride_scale_dim: tl.constexpr,
                            BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_HWM: tl.constexpr,
                            BLOCK_SIZE_HEAD: tl.constexpr, BLOCK_SIZE_HEAD_DIV_NUM_SMS: tl.constexpr,
                            NUM_SMS: tl.constexpr):
    # Triton kernel
