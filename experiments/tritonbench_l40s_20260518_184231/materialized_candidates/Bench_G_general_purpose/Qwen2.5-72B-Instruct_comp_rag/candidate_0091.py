import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Logics_ptr, V_ptr, Out_ptr, B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
                stride_logic_b, stride_logic_h, stride_logic_n,
                stride_v_b, stride_v_h, stride_v_n,
                stride_out_b, stride_out_h, stride_out_n,
                BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load metadata
    b_loc = tl.load(B_Loc_ptr + pid_b)
    b_start_loc = tl.load(B_Start_Loc_ptr + pid_b)
    b_seqlen = tl.load(B_Seqlen_ptr + pid_b)

    # Initialize variables
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    e_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    e_sum = tl.zeros((1,), dtype=tl.float32)

    for start_n in range(0, b_seqlen, BLOCK_N):
        end_n = min(start_n + BLOCK_N, b_seqlen)
        n = end_n - start_n

        # Load logits and V
        logits_ptrs = Logics_ptr + b_loc * stride_logic_b + pid_h * stride_logic_h + start_n * stride_logic_n
        v_ptrs = V_ptr + b_loc * stride_v_b + pid_h * stride_v_h + start_n * stride_v_n

        logits = tl.load(logits_ptrs, mask=start_n + tl.arange(0, BLOCK_N) < b_seqlen, other=-float('inf'))
        v = tl.load(v_ptrs, mask=start_n + tl.arange(0, BLOCK_N) < b_seqlen, other=0.0)

        # Compute max
        e_max = tl.maximum(e_max, tl.max(logits, axis=0))

        # Compute exponentials
        logits = logits - e_max
        p = tl.exp(logits)

        # Compute sum of exponentials
        e_sum += tl.sum(p, axis=0)

        # Compute weighted sum
        acc += tl.dot(p, v)

    # Normalize by sum of exponentials
    acc /= e_sum

    # Store results
    out_ptr = Out_ptr + b_loc * stride_out_b + pid_h * stride_out_h
    tl.store(out_ptr, acc, mask=tl.arange(0, BLOCK_DMODEL) < BLOCK_DMODEL)

def token_softmax_reducev_fwd(Logics, V, B_Loc, B_Start_Loc, B_Seqlen, Out, BLOCK_DMODEL, BLOCK_N):
    # Get tensor dimensions
    B, H, N = Logics.shape
    _, _, D = V.shape

    # Define grid and block sizes
    grid = (B, H, 1)
    num_warps = 4
    num_stages = 2

    # Launch kernel
    _fwd_kernel[grid](
        Logics, V, Out, B_Loc, B_Start_Loc, B_Seqlen,
        Logics.stride(0), Logics.stride(1), Logics.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages
    )

torch.manual_seed(42)

# Example input tensors
B, H, N, D = 2, 4, 128, 64
Logics = torch.randn(B, H, N, device='cuda')
V = torch.randn(B, H, N, D, device='cuda')
B_Loc = torch.arange(B, device='cuda')
B_Start_Loc = torch.zeros(B, dtype=torch.int32, device='cuda')
B_Seqlen = torch.full((B,), N, dtype=torch.int32, device='cuda')
Out = torch.empty(B, H, D, device='cuda')

# Define block sizes
BLOCK_DMODEL = D
BLOCK_N = 32

# Call the wrapper function
token_softmax_reducev_fwd(Logics, V, B_Loc, B_Start_Loc, B_Seqlen, Out, BLOCK_DMODEL, BLOCK_N)

# Print the result
print(Out)
