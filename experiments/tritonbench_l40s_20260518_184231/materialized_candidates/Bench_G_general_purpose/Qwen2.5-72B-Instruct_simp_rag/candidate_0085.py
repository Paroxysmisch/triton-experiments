import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Out, Logics, V, B_Loc, B_Start_Loc, B_Seqlen,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    stride_logics_b, stride_logics_h, stride_logics_s,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_b_loc_b, stride_b_loc_h,
    stride_b_start_loc_b,
    stride_b_seqlen_b,
    batch, head, seq_len, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid // head
    hid = pid % head

    b_loc = tl.load(B_Loc + bid * stride_b_loc_b + hid * stride_b_loc_h)
    b_start_loc = tl.load(B_Start_Loc + bid * stride_b_start_loc_b)
    b_seqlen = tl.load(B_Seqlen + bid * stride_b_seqlen_b)

    start_n = b_start_loc + tl.program_id(1) * BLOCK_SIZE
    offs_n = start_n + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < b_seqlen

    offs_d = tl.arange(0, dim)
    offs_b = bid * stride_logics_b
    offs_h = hid * stride_logics_h

    for n in range(0, b_seqlen, BLOCK_SIZE):
        logics_ptr = Logics + offs_b + offs_h + n * stride_logics_s + offs_d
        v_ptr = V + offs_b + hid * stride_v_h + n * stride_v_s + offs_d
        out_ptr = Out + bid * stride_out_b + hid * stride_out_h + n * stride_out_s + offs_d

        logics = tl.load(logics_ptr, mask=mask, other=-float('inf'))
        v = tl.load(v_ptr, mask=mask, other=0.0)

        max_logics = tl.max(logics, axis=0)
        logics_minus_max = logics - max_logics
        exp_logics = tl.exp(logics_minus_max)
        sum_exp_logics = tl.sum(exp_logics, axis=0)
        softmax_output = exp_logics / sum_exp_logics

        out = tl.sum(softmax_output[:, None] * v, axis=0)
        tl.store(out_ptr, out, mask=mask)

def token_softmax_reducev_fwd(Out, Logics, V, B_Loc, B_Start_Loc, B_Seqlen, batch, head, seq_len, dim):
    BLOCK_SIZE = 128
    num_programs = batch * head
    grid = (num_programs, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)

    _fwd_kernel[grid](
        Out, Logics, V, B_Loc, B_Start_Loc, B_Seqlen,
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Logics.stride(0), Logics.stride(1), Logics.stride(2),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B_Loc.stride(0), B_Loc.stride(1),
        B_Start_Loc.stride(0),
        B_Seqlen.stride(0),
        batch, head, seq_len, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example tensors
batch = 2
head = 4
seq_len = 128
dim = 64

Logics = torch.randn(batch, head, seq_len, dim, device='cuda')
V = torch.randn(batch, head, seq_len, dim, device='cuda')
Out = torch.empty_like(V)
B_Loc = torch.arange(batch * head, device='cuda').view(batch, head)
B_Start_Loc = torch.zeros(batch, device='cuda')
B_Seqlen = torch.full((batch,), seq_len, device='cuda')

# Call the forward function
token_softmax_reducev_fwd(Out, Logics, V, B_Loc, B_Start_Loc, B_Seqlen, batch, head, seq_len, dim)

# Print the output
print(Out)
