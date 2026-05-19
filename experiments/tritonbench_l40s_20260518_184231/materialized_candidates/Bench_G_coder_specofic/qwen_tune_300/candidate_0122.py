import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

@triton.jit
def _rope_embedding(
    Q: tl.tensor,
    Q_row_stride: int,
    cos: tl.tensor,
    sin: tl.tensor,
    seqlen: int,
    head_dim: int,
    n_heads: int,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute the RoPE embedding. This is based on the implementation in the
    FSDP tutorial: https://pytorch.org/tutorials/recipes/functional_stateful.html
    """
    row_position = tl.program_id(0)
    half_head_dim = head_dim // 2
    # The block size is half the head dimension in this case, since we process
    # two halves of the head at the same time.
    block_position = tl.program_id(1) * BLOCK_SIZE

    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = Q + row_position * Q_row_stride
    # We also advance the pointer by half the head dimension to process the other half
    row_start_ptr_other_half = row_start_ptr + half_head_dim

    cos_start_ptr = cos + row_position * half_head_dim
    sin_start_ptr = sin + row_position * half_head_dim

    # In the backward pass, we also pass the original Q tensor as an argument since
    # we need it to compute the gradients.
    if BACKWARD_PASS:
        Q_transposed = tl.tensor(
            tl.trans(Q),
            dtype=tl.float32,
        )
        q0 = tl.load(
            Q_transposed + half_head_dim * n_heads,
            mask=(row_position < seqlen) & (half_head_dim == BLOCK_SIZE),
            other=0.0,
        )
        q0_grad = tl.load(
            Q_transposed + half_head_dim * n_heads,
            mask=(row_position < seqlen) & (half_head_dim == BLOCK_SIZE),
            other=0.0,
        )
    else:
        q0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in range(0, BLOCK_SIZE, 1):
        block_start_k = block_position + k
        # Advance the pointer to the start of the block for the current half-head
        q_k_ptr = row_start_ptr + block_start_k
        q_k_other_half_ptr = row_start_ptr_other_half + block_start_k

        cos_k_ptr = cos_start_ptr + block_start_k
        sin_k_ptr = sin_start_ptr + block_start_k

        # Load the values for the current block element. We cast to float32 here to
        # avoid some mysterious overflow issues that occur if we leave it as int32.
        q_k = tl.load(q_k_ptr).to(tl.float32)
        q_k_other_half = tl.load(q_k_other_half_ptr).to(tl.float32)
        cos_k = tl.load(cos_k_ptr).to(tl.float32)
        sin_k = tl.load(sin_k_ptr).to(tl.float32)

        # This is the most memory-efficient way we could think of to compute the
        # half-transformations in parallel. We load the two halves into two vectors,
        # perform the transformations independently, and then write them back out.
        q_k_rotated = q_k * cos_k - q_k_other_half * sin_k
        q_k_other_half_rotated = q_k * sin_k + q_k_other_half * cos_k

        # We only write the first half back to the output tensor. The second half
        # will be written later.
        tl.store(q_k_ptr, q_k_rotated.to(Q.dtype.element_ty))
        tl.store(q_k_other_half_ptr, q_k_other_half_rotated.to(Q.dtype.element_ty))

        if BACKWARD_PASS:
            # In the backward pass, we load the gradients from the gradients tensor
            q_k_grad = tl.load(q_k_ptr).to(tl.float32)
            q_k_other_half_grad = tl.load(q_k_other_half_ptr).to(tl.float32)

            # We then compute the gradients for Q
            q0_grad -= q_k_grad * cos_k + q_k_other_half_grad * sin_k
            q0_grad -= q_k_other_half_grad * cos_k + q_k_grad * sin_k

    if BACKWARD_PASS:
        # Write the gradients for Q to the appropriate place in the gradients tensor
        tl.store(
            q0,
            q0_grad.to(Q.dtype.element_ty),
        )
    else:
        # For the forward pass, we also write the transformed second half back to the
        # output tensor.
        half_head_dim_ptr = row_start_ptr_other_half
        half_head_dim_other_half_ptr = row_start_ptr

        cos_half_head_dim_ptr = cos_start_ptr
        sin_half_head_dim_ptr = sin_start_ptr

        for k in range(0, BLOCK_SIZE, 1):
            block_start_k = block_position + k

            q_k_other_half_ptr = half_head_dim_ptr + block_start_k
            q_k_ptr = half_head_dim_other_half_ptr + block_start_k

            cos_k_ptr = cos_half_head_dim_ptr + block_start_k
            sin_k_ptr = sin_half_head_dim_ptr + block_start_k

            q_k_other_half = tl.load(q_k_other_half_ptr).to(tl.float32)
            q_k = tl.load(q_k_ptr).to(tl.float32)
            cos_k = tl.load(cos_k_ptr).to(tl.float32)
            sin_k = tl.load(sin_k_ptr).to(tl.float32)

            q_k_rotated = q_k * cos_k - q_k_other_half * sin_k
            q_k_other_half_rotated = q_k * sin_k + q_k_other_half * cos_k

            tl.store(q_k_other_half_ptr, q_k_rotated.to(Q.dtype.element_ty))
            tl.store(q_k_ptr, q_k_other_half_rotated.to(Q.dtype.element_ty))


class Fast_RoPE_Embedding(Function):
    @staticmethod
    def forward(ctx, q, cos, sin):
        # Transpose the query matrix since the rest of the code works with flattened
        # versions of matrices rather than actual 2D matrices.
        q = q.transpose(1, 2)
        M, N = q.shape
        q_reshaped = q.reshape(M * N, -1)
        output = torch.empty_like(q_reshaped)

        BLOCK_SIZE = triton.next_power_of_2(N // 2)

        # The backward pass is slightly simpler since we don't need to swap the two halves
        # of the head back.
        def grid(META):
            return (triton.cdiv(M * N, META["BLOCK_SIZE"]), M)

        # We need to make sure to backprop through the cos and sin values as well since
        # they are learnable parameters in the model.
        cos, sin = Fast_RoPE_Embedding._enable_grad(cos), Fast_RoPE_Embedding._enable_grad(sin)
        with torch.cuda.device(q.device):
            _rope_embedding[grid](q_reshaped, q.stride(0), cos, sin, M, N, BLOCK_SIZE=BLOCK_SIZE)
        output = q.reshape(M, N).transpose(1, 2)
        ctx.save_for_backward(cos, sin)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.transpose(1, 2)
        M, N = grad_output.shape
        grad_output_reshaped = grad_output.reshape(M * N, -1)
        output = torch.empty_like(grad_output_reshaped)

        cos, sin = ctx.saved_tensors

        BLOCK_SIZE = triton.next_power_of_2(N // 2)

        def grid(META):
            return (triton.cdiv(M * N, META["BLOCK_SIZE"]), M)

        with torch.cuda.device(grad_output.device):
            _rope_embedding[grid](
                grad_output_reshaped,
                grad_output.stride(0),
                cos,
                sin,
                M,
                N,
                BACKWARD_PASS=True,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        output = grad_output.reshape(M, N).transpose(1, 2)
        return output, None, None

    @staticmethod
    def _enable_grad(tensor):
        if tensor.requires_grad is False:
            return tensor.detach().requires_grad_(True)
        else:
            return tensor


def fast_rope_embedding(q: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Compute the RoPE embedding in a fast, parallel-friendly way. This implementation is
    based on the FSDP tutorial and should work with arbitrary sequence lengths and head
    dimensions.
    """
    return Fast_RoPE_Embedding.apply(q, cos, sin)
