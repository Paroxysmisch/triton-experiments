import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_transformer_block_kernel(
    # Pointers to matrices
    input_ptr, weight1_ptr, weight2_ptr, residual_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, d_in, d_k, d_out,
    # Strides for the different matrices
    input_batch_stride, input_row_stride, input_col_stride,
    weight1_stride, weight2_stride,
    residual_batch_stride, residual_row_stride,
    output_batch_stride, output_row_stride,
    # Other parameters
    dropout_p, eps,
    # Random seed for dropout
    seed,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(d_k, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    local_pid = pid % num_pid_in_group
    pid_m = local_pid // num_pid_n
    pid_n = local_pid % num_pid_n

    # Block pointers
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    block_k = 0

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Pointers to current blocks
    a_ptr = input_ptr + group_id * input_batch_stride + block_m * input_row_stride
    b_ptr = weight1_ptr + block_n * weight1_stride

    # Load blocks from A (input) and B (weight1)
    for k in range(0, d_in, BLOCK_SIZE_K):
        a = tl.load(a_ptr + k)
        b = tl.load(b_ptr + k)
        acc += tl.dot(a, b)

    # Apply softmax
    acc = tl.softmax(acc)

    # Apply dropout
    if dropout_p > 0.0:
        rand = tl.rand(seed, acc.shape)
        acc = tl.where(rand > dropout_p, acc / (1.0 - dropout_p), 0.0)

    # Second matrix multiplication with weight2
    c_ptr = weight2_ptr
    result = tl.dot(acc, tl.load(c_ptr))

    # Add residual
    if residual_ptr is not None:
        residual = tl.load(residual_ptr + group_id * residual_batch_stride + block_m * residual_row_stride)
        result += residual

    # Layer normalization
    mean = tl.mean(result, axis=1, keepdims=True)
    var = tl.var(result, axis=1, keepdims=True)
    result = (result - mean) / tl.sqrt(var + eps)

    # Store result
    output_ptr = output_ptr + group_id * output_batch_stride + block_m * output_row_stride
    tl.store(output_ptr, result)

class FusedTransformerBlock(torch.nn.Module):
    def __init__(self, d_in, d_k, d_out, dropout_p=0.1, eps=1e-5):
        super().__init__()
        self.d_in = d_in
        self.d_k = d_k
        self.d_out = d_out
        self.dropout_p = dropout_p
        self.eps = eps
        
        # Learnable parameters
        self.weight1 = torch.nn.Parameter(torch.empty(d_in, d_k))
        self.weight2 = torch.nn.Parameter(torch.empty(d_k, d_out))
        self.gamma = torch.nn.Parameter(torch.ones(d_out))
        self.beta = torch.nn.Parameter(torch.zeros(d_out))
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, input, residual=None, dropout_p=None, eps=None, out=None):
        # Handle default parameters
        dropout_p = dropout_p if dropout_p is not None else self.dropout_p
        eps = eps if eps is not None else self.eps
        
        # Get input dimensions
        *batch_dims, seq_len, _ = input.shape
        batch_size = math.prod(batch_dims)
        
        # Prepare output tensor
        if out is None:
            out = torch.empty(*batch_dims, seq_len, self.d_out, device=input.device, dtype=input.dtype)
        
        # Configure block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 16
        
        # Launch kernel
        grid = (batch_size * triton.cdiv(seq_len, BLOCK_SIZE_M) * triton.cdiv(self.d_k, BLOCK_SIZE_N),)
        
        fused_transformer_block_kernel[grid](
            input.contiguous(), self.weight1, self.weight2,
            residual.contiguous() if residual is not None else None,
            out,
            batch_size, seq_len, self.d_in, self.d_k, self.d_out,
            input.stride(0), input.stride(1), input.stride(2),
            self.weight1.stride(0), self.weight2.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual.stride(1) if residual is not None else 0,
            out.stride(0), out.stride(1),
            dropout_p, eps,
            torch.randint(0, 2**31-1, (1,)).item(),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        
        return out
