import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_recurrence(
    # Pointers to tensors
    S_ptr, d_ptr, O_ptr, last_kv_ptr,
    # Dimensions
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    stride_s_h, stride_s_b, stride_s_d,
    stride_d_h, stride_d_b,
    stride_o_h, stride_o_b, stride_o_d,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    # Compute offsets for this head and block
    s_offset = head_id * stride_s_h + block_id * stride_s_b
    d_offset = head_id * stride_d_h + block_id * stride_d_b
    o_offset = head_id * stride_o_h + block_id * stride_o_b

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load previous state if available
    if last_kv_ptr != 0:
        last_kv = tl.load(last_kv_ptr + head_id * BLOCK_SIZE + 
                         tl.arange(0, BLOCK_SIZE))
        acc += last_kv

    # Main recurrence loop
    for i in range(0, D_MODEL_K, BLOCK_SIZE):
        # Load block of S and d
        block_range = tl.arange(0, BLOCK_SIZE) + i
        mask = block_range < D_MODEL_K
        s = tl.load(S_ptr + s_offset + block_range, mask=mask)
        d = tl.load(d_ptr + d_offset + block_range, mask=mask)
        
        # Update accumulator
        acc = acc * d + s

        # Store result
        tl.store(O_ptr + o_offset + block_range, acc, mask=mask)


@triton.jit
def _bwd_recurrence(
    # Pointers to tensors
    S_ptr, d_ptr, DI_ptr, DG_ptr, DL_ptr, DS_ptr,
    # Dimensions
    NUM_HEAD: tl.constexpr, NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, BLOCK_MODEL_V: tl.constexpr,
    stride_s_h, stride_s_b, stride_s_d,
    stride_d_h, stride_d_b,
    stride_di_h, stride_di_b, stride_di_d,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    head_id = pid // NUM_BLOCK
    block_id = pid % NUM_BLOCK

    # Compute offsets
    s_offset = head_id * stride_s_h + block_id * stride_s_b
    d_offset = head_id * stride_d_h + block_id * stride_d_b
    di_offset = head_id * stride_di_h + block_id * stride_di_b

    # Initialize gradient accumulators
    d_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    s_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Backward pass loop
    for i in range(D_MODEL_K-BLOCK_SIZE, -1, -BLOCK_SIZE):
        block_range = tl.arange(0, BLOCK_SIZE) + i
        mask = block_range < D_MODEL_K
        
        # Load gradients and values
        di = tl.load(DI_ptr + di_offset + block_range, mask=mask)
        s = tl.load(S_ptr + s_offset + block_range, mask=mask)
        d = tl.load(d_ptr + d_offset + block_range, mask=mask)

        # Compute gradients
        d_acc = d_acc * d + di
        s_acc = s_acc + di

        # Store gradients
        tl.store(DG_ptr + d_offset + block_range, d_acc, mask=mask)
        tl.store(DS_ptr + s_offset + block_range, s_acc, mask=mask)

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        # Get dimensions
        batch_size, num_heads, seq_len, d_model = kv.shape
        
        # Configure block sizes
        BLOCK_MODEL_K = 32
        BLOCK_MODEL_V = 32
        NUM_BLOCK = (seq_len + BLOCK_MODEL_K - 1) // BLOCK_MODEL_K
        
        # Allocate output tensor
        output = torch.empty_like(kv)
        
        # Prepare last_kv pointer
        last_kv_ptr = 0 if last_kv is None else last_kv.data_ptr()
        
        # Launch kernel
        grid = (num_heads * NUM_BLOCK,)
        _fwd_recurrence[grid](
            kv.data_ptr(), cross_decay.data_ptr(), output.data_ptr(),
            last_kv_ptr,
            num_heads, NUM_BLOCK,
            d_model, d_model,
            BLOCK_MODEL_K, BLOCK_MODEL_V,
            kv.stride(1), kv.stride(2), kv.stride(3),
            cross_decay.stride(1), cross_decay.stride(2),
            output.stride(1), output.stride(2), output.stride(3),
            BLOCK_MODEL_K
        )
        
        # Save for backward
        ctx.save_for_backward(kv, cross_decay, output)
        ctx.dimensions = (num_heads, NUM_BLOCK, d_model)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kv, cross_decay, output = ctx.saved_tensors
        num_heads, NUM_BLOCK, d_model = ctx.dimensions
        
        # Allocate gradient tensors
        grad_kv = torch.empty_like(kv)
        grad_decay = torch.empty_like(cross_decay)
        
        # Launch backward kernel
        grid = (num_heads * NUM_BLOCK,)
        _bwd_recurrence[grid](
            kv.data_ptr(), cross_decay.data_ptr(),
            grad_output.data_ptr(), grad_decay.data_ptr(),
            torch.zeros_like(output).data_ptr(), grad_kv.data_ptr(),
            num_heads, NUM_BLOCK,
            d_model, d_model,
            32, 32,  # BLOCK_MODEL_K, BLOCK_MODEL_V
            kv.stride(1), kv.stride(2), kv.stride(3),
            cross_decay.stride(1), cross_decay.stride(2),
            grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            32  # BLOCK_SIZE
        )
        
        return grad_kv, grad_decay, None
