import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    input1_ptr, input2_ptr, other_ptr, mat2_ptr, output_ptr,
    # Tensor dimensions
    B, M, N, D_in, D_out,
    # Parameters
    p, training: tl.constexpr, dim: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Element-wise operations
    input1 = tl.load(input1_ptr + pid_b * M * N + pid_m * N + pid_n)
    input2 = tl.load(input2_ptr + pid_b * M * N + pid_m * N + pid_n)
    z = input1 * input2
    
    # Handle broadcasting for 'other'
    other = tl.load(other_ptr + (pid_n if other_ptr != 0 else 0))
    s = z + other

    # Log-softmax computation
    max_val = tl.max(s, axis=dim)
    exp_s = tl.exp(s - max_val)
    sum_exp = tl.sum(exp_s, axis=dim)
    log_softmax = (s - max_val) - tl.log(sum_exp)

    # Dropout application
    if training:
        mask = tl.rand(tl.float32) > p
        dropout = tl.where(mask, log_softmax / (1 - p), 0.0)
    else:
        dropout = log_softmax

    # Batch matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    for d in range(0, D_in):
        a = tl.load(dropout + pid_b * M * D_in + pid_m * D_in + d)
        b = tl.load(mat2_ptr + pid_b * D_in * D_out + d * D_out + pid_n)
        acc += a * b

    # Store final result
    tl.store(output_ptr + pid_b * M * D_out + pid_m * D_out + pid_n, acc)

def fused_mul_add_logsoftmax_dropout_bmm(
    input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None
):
    # Shape validation and broadcasting
    assert input1.dim() >= 2 and mat2.dim() == 3
    B, M, N = input1.shape[:-1], input1.size(-1)
    D_in, D_out = mat2.shape[-2], mat2.shape[-1]
    
    # Kernel configuration
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(D_out, meta['BLOCK_SIZE_D']))
    
    # Allocate output tensor
    output = torch.empty((B, M, D_out), device=input1.device)
    
    # Launch kernel
    fused_kernel[grid](
        input1, input2, other, mat2, output,
        B, M, N, D_in, D_out,
        p, training, dim,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_D=32
    )
    
    return output if out is None else out.copy_(output)
