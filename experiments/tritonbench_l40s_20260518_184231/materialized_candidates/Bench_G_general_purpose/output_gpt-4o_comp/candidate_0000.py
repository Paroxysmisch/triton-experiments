import torch
import triton
import triton.language as tl

# Define block sizes
BLOCK = 64
CBLOCK = 32

# Forward kernel
@triton.jit
def _fwd_kernel(Q, K, V, Out, stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                stride_vz, stride_vh, stride_vn, stride_vk,
                stride_oz, stride_oh, stride_om, stride_ok,
                BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_block = tl.cdiv(stride_qm, BLOCK)
    
    # Compute batch and head indices
    batch_id = pid // num_block
    block_id = pid % num_block
    
    # Load Q, K, V for this block
    q_ptrs = Q + batch_id * stride_qz + block_id * BLOCK * stride_qm
    k_ptrs = K + batch_id * stride_kz + block_id * BLOCK * stride_kn
    v_ptrs = V + batch_id * stride_vz + block_id * BLOCK * stride_vn
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    for i in range(num_block):
        # Load segments of Q, K, V
        q = tl.load(q_ptrs + i * BLOCK * stride_qk)
        k = tl.load(k_ptrs + i * BLOCK * stride_kk)
        v = tl.load(v_ptrs + i * BLOCK * stride_vk)
        
        # Compute dot product
        dot = tl.dot(q, k)
        
        # Update accumulator with attention result
        acc += tl.dot(dot, v)
    
    # Write output
    out_ptrs = Out + batch_id * stride_oz + block_id * BLOCK * stride_om
    tl.store(out_ptrs, acc)

# Backward intra-block kernel
@triton.jit
def _bwd_intra_kernel(DO, Q, K, V, DQ, DK, DV, stride_qz, stride_qh, stride_qm, stride_qk,
                      stride_kz, stride_kh, stride_kn, stride_kk,
                      stride_vz, stride_vh, stride_vn, stride_vk,
                      stride_doz, stride_doh, stride_dom, stride_dok,
                      BLOCK: tl.constexpr, CBLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_block = tl.cdiv(stride_qm, BLOCK)
    
    # Compute batch and head indices
    batch_id = pid // num_block
    block_id = pid % num_block
    
    # Load DO, Q, K, V for this block
    do_ptrs = DO + batch_id * stride_doz + block_id * BLOCK * stride_dom
    q_ptrs = Q + batch_id * stride_qz + block_id * BLOCK * stride_qm
    k_ptrs = K + batch_id * stride_kz + block_id * BLOCK * stride_kn
    v_ptrs = V + batch_id * stride_vz + block_id * BLOCK * stride_vn
    
    # Initialize gradients
    dq = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    dv = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    for i in range(num_block):
        # Load segments of DO, Q, K, V
        do = tl.load(do_ptrs + i * BLOCK * stride_dok)
        q = tl.load(q_ptrs + i * BLOCK * stride_qk)
        k = tl.load(k_ptrs + i * BLOCK * stride_kk)
        v = tl.load(v_ptrs + i * BLOCK * stride_vk)
        
        # Compute gradients
        dq += tl.dot(do, k)
        dk += tl.dot(q, do)
        dv += tl.dot(do, v)
    
    # Write gradients
    dq_ptrs = DQ + batch_id * stride_qz + block_id * BLOCK * stride_qm
    dk_ptrs = DK + batch_id * stride_kz + block_id * BLOCK * stride_kn
    dv_ptrs = DV + batch_id * stride_vz + block_id * BLOCK * stride_vn
    
    tl.store(dq_ptrs, dq)
    tl.store(dk_ptrs, dk)
    tl.store(dv_ptrs, dv)

# Backward inter-block kernel
@triton.jit
def _bwd_inter_kernel(DO, Q, K, V, DQ, DK, DV, stride_qz, stride_qh, stride_qm, stride_qk,
                      stride_kz, stride_kh, stride_kn, stride_kk,
                      stride_vz, stride_vh, stride_vn, stride_vk,
                      stride_doz, stride_doh, stride_dom, stride_dok,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_block = tl.cdiv(stride_qm, BLOCK)
    
    # Compute batch and head indices
    batch_id = pid // num_block
    
    # Initialize accumulators
    acc_dk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    
    for block_id in range(num_block):
        # Load DO, Q, K, V for this block
        do_ptrs = DO + batch_id * stride_doz + block_id * BLOCK * stride_dom
        q_ptrs = Q + batch_id * stride_qz + block_id * BLOCK * stride_qm
        k_ptrs = K + batch_id * stride_kz + block_id * BLOCK * stride_kn
        v_ptrs = V + batch_id * stride_vz + block_id * BLOCK * stride_vn
        
        do = tl.load(do_ptrs)
        q = tl.load(q_ptrs)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Update accumulators with inter-block interactions
        acc_dk += tl.dot(q, do)
        acc_dv += tl.dot(do, v)
    
    # Write accumulated gradients
    dk_ptrs = DK + batch_id * stride_kz
    dv_ptrs = DV + batch_id * stride_vz
    
    tl.store(dk_ptrs, acc_dk)
    tl.store(dv_ptrs, acc_dv)
