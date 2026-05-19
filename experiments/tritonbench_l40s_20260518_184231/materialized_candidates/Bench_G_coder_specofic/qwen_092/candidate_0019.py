import triton
import triton.language as tl

# Triton kernel function to compute softmax probabilities from logits
@triton.jit
def _fwd_kernel_token_softmax(
    Logits: tl.tensor,  # [B, H, N]
    B_Start_Loc: tl.tensor,  # [B]
    B_Seqlen: tl.tensor,  # [B]
    Prob_Out: tl.tensor,  # [B, H, N]
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the current block
    pid = tl.program_id(axis=0)
    # Get the block ID for the current warp
    bid = tl.program_id(axis=1)
    # Get the global ID within the grid
    gid = pid * BLOCK_SIZE + bid
    # Get the batch and head indices
    b_idx = gid // (BLOCK_SIZE * Logits.shape[1])
    h_idx = (gid // BLOCK_SIZE) % Logits.shape[1]
    # Get the sequence length for the current batch
    seq_len = B_Seqlen[b_idx]
    # Calculate the start index for the current batch
    start_idx = B_Start_Loc[b_idx]
    # Calculate the end index for the current batch
    end_idx = start_idx + seq_len
    # Load the logits for the current batch and head
    logits = Logits[b_idx, h_idx, start_idx:end_idx]
    # Compute the maximum value for numerical stability
    max_val = tl.max(logits)
    # Subtract the maximum value from the logits
    logits = logits - max_val
    # Compute the exponentials of the logits
    exps = tl.exp(logits)
    # Compute the sum of the exponentials
    sum_exps = tl.sum(exps)
    # Compute the softmax probabilities
    softmax_probs = exps / sum_exps
    # Store the softmax probabilities in the output tensor
    Prob_Out[b_idx, h_idx, start_idx:end_idx] = softmax_probs

# Wrapper function to launch the Triton kernel
@torch.no_grad()
def token_softmax_fwd(
    Logits: torch.Tensor,  # [B, H, N]
    B_Start_Loc: torch.Tensor,  # [B]
    B_Seqlen: torch.Tensor,  # [B]
    Prob_Out: torch.Tensor,  # [B, H, N]
    max_input_len: int,
):
    # Determine the block size based on the maximum input length
    BLOCK_SIZE = 256
    # Calculate the number of warps needed
    num_warps = 4
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(Logits.shape[0] * Logits.shape[1], BLOCK_SIZE),)
    _fwd_kernel_token_softmax[grid, (BLOCK_SIZE, num_warps, 1)](
        Logits,
        B_Start_Loc,
        B_Seqlen,
        Prob_Out,
        BLOCK_SIZE,
    )
