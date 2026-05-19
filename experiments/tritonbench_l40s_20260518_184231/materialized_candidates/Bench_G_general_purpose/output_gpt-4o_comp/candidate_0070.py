import torch
import triton
import triton.language as tl

# Kernel for forward pass to compute hidden states
@triton.jit
def chunk_retention_fwd_kernel_h(k_ptr, v_ptr, h_ptr, initial_state_ptr, final_state_ptr, 
                                BT, boundary_check, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load k and v
    k = tl.load(k_ptr + offsets, mask=offsets < BT, other=0.0)
    v = tl.load(v_ptr + offsets, mask=offsets < BT, other=0.0)
    
    # Load initial state
    initial_state = tl.load(initial_state_ptr + offsets, mask=offsets < BT, other=0.0)
    
    # Compute hidden state
    h = initial_state + k * v
    tl.store(h_ptr + offsets, h, mask=offsets < BT)
    
    # Store final state if needed
    if boundary_check:
        tl.store(final_state_ptr + offsets, h, mask=offsets < BT)

# Kernel for forward pass to compute final output
@triton.jit
def chunk_retention_fwd_kernel_o(q_ptr, k_ptr, v_ptr, output_ptr, scale, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load q, k, v
    q = tl.load(q_ptr + offsets, mask=offsets < BT, other=0.0)
    k = tl.load(k_ptr + offsets, mask=offsets < BT, other=0.0)
    v = tl.load(v_ptr + offsets, mask=offsets < BT, other=0.0)
    
    # Compute output
    output = q * (k * v) * scale
    tl.store(output_ptr + offsets, output, mask=offsets < BT)

# Kernel for backward pass to compute gradients for hidden states
@triton.jit
def chunk_retention_bwd_kernel_dh(q_ptr, do_ptr, dh_ptr, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load q and do
    q = tl.load(q_ptr + offsets, mask=offsets < BT, other=0.0)
    do = tl.load(do_ptr + offsets, mask=offsets < BT, other=0.0)
    
    # Compute gradient for hidden state
    dh = q * do
    tl.store(dh_ptr + offsets, dh, mask=offsets < BT)

# Kernel for backward pass to compute gradients for q, k, v
@triton.jit
def chunk_retention_bwd_kernel_dqkv(do_ptr, dh_ptr, h_ptr, dq_ptr, dk_ptr, dv_ptr, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load do, dh, h
    do = tl.load(do_ptr + offsets, mask=offsets < BT, other=0.0)
    dh = tl.load(dh_ptr + offsets, mask=offsets < BT, other=0.0)
    h = tl.load(h_ptr + offsets, mask=offsets < BT, other=0.0)
    
    # Compute gradients for q, k, v
    dq = do * h
    dk = do * dh
    dv = do * dh
    
    tl.store(dq_ptr + offsets, dq, mask=offsets < BT)
    tl.store(dk_ptr + offsets, dk, mask=offsets < BT)
    tl.store(dv_ptr + offsets, dv, mask=offsets < BT)

# Python wrapper functions
def chunk_fwd_h_fn(k, v, initial_state, BT, boundary_check):
    BLOCK_SIZE = 128  # example block size
    h = torch.empty_like(k)
    final_state = torch.empty_like(initial_state)
    grid = (BT // BLOCK_SIZE,)
    chunk_retention_fwd_kernel_h[grid](k, v, h, initial_state, final_state, BT, boundary_check, BLOCK_SIZE=BLOCK_SIZE)
    return h, final_state

def chunk_fwd_o_fn(q, k, v, scale, BT):
    BLOCK_SIZE = 128  # example block size
    output = torch.empty_like(q)
    grid = (BT // BLOCK_SIZE,)
    chunk_retention_fwd_kernel_o[grid](q, k, v, output, scale, BT, BLOCK_SIZE=BLOCK_SIZE)
    return output

def chunk_bwd_dh_fn(q, do, BT):
    BLOCK_SIZE = 128  # example block size
    dh = torch.empty_like(q)
    grid = (BT // BLOCK_SIZE,)
    chunk_retention_bwd_kernel_dh[grid](q, do, dh, BT, BLOCK_SIZE=BLOCK_SIZE)
    return dh

def chunk_bwd_dqkv_fn(do, dh, h, BT):
    BLOCK_SIZE = 128  # example block size
    dq = torch.empty_like(do)
    dk = torch.empty_like(do)
    dv = torch.empty_like(do)
    grid = (BT // BLOCK_SIZE,)
    chunk_retention_bwd_kernel_dqkv[grid](do, dh, h, dq, dk, dv, BT, BLOCK_SIZE=BLOCK_SIZE)
    return dq, dk, dv

# PyTorch autograd function
class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state, scale, BT, boundary_check):
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.scale = scale
        ctx.BT = BT
        ctx.boundary_check = boundary_check
        
        h, final_state = chunk_fwd_h_fn(k, v, initial_state, BT, boundary_check)
        output = chunk_fwd_o_fn(q, k, v, scale, BT)
        ctx.h = h
        
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, initial_state = ctx.saved_tensors
        h = ctx.h
        scale = ctx.scale
        BT = ctx.BT
        boundary_check = ctx.boundary_check
        
        dh = chunk_bwd_dh_fn(q, grad_output, BT)
        dq, dk, dv = chunk_bwd_dqkv_fn(grad_output, dh, h, BT)
        
        return dq, dk, dv, None, None, None, None

# User-friendly interface
def chunk_retention(q, k, v, initial_state=None, scale=1.0, boundary_check=False):
    BT = q.size(0)  # assuming batch size is the first dimension
    if initial_state is None:
        initial_state = torch.zeros_like(q)
    
    output, final_state = ChunkRetentionFunction.apply(q, k, v, initial_state, scale, BT, boundary_check)
    return output, final_state
