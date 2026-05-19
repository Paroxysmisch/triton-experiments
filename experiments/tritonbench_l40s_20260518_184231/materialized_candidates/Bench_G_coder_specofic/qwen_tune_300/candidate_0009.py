import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform

if triton.__version__ >= "2.1.0":

    @triton.jit
    def calculate_settings(n):
        # Determine BLOCK_SIZE and num_warps heuristically
        BLOCK_SIZE = 2
        while BLOCK_SIZE <= n:
            if current_platform.is_power_of_two(BLOCK_SIZE):
                num_warps = 2
                while num_warps * BLOCK_SIZE <= 1024:
                    if (1024 // (num_warps * BLOCK_SIZE)) <= 64:
                        return BLOCK_SIZE, num_warps
                    num_warps *= 2
            BLOCK_SIZE *= 2
        raise RuntimeError("BLOCK_SIZE exceeds MAX_FUSED_SIZE")

    @triton.jit
    def _rope_embedding(Q, Q_row_stride, cos, cos_row_stride, sin, sin_row_stride, seqlen, head_dim, n_heads,
                         BACKWARD_PASS, BLOCK_SIZE: tl.constexpr, ROPE_GROUP_SIZE: tl.constexpr):
        # Triton kernel for RoPE embedding computation
        row_position = tl.program_id(0)
        Q_row_start = Q + row_position * Q_row_stride
        cos_row_start = cos + row_position * cos_row_stride
        sin_row_start = sin + row_position * sin_row_stride

        # q_1, q_2, ... q_{d_{head}}
        q = tl.load(Q_row_start + tl.arange(0, BLOCK_SIZE))
        q_cos = tl.load(cos_row_start + tl.arange(0, BLOCK_SIZE))
        q_sin = tl.load(sin_row_start + tl.arange(0, BLOCK_SIZE))

        if not BACKWARD_PASS:
            q_upper = q
            q_lower = q
            q_cos_upper = q_cos
            q_cos_lower = q_cos
            q_sin_upper = q_sin
            q_sin_lower = q_sin
        else:
            q_upper = tl.load(Q_row_start + tl.arange(0, BLOCK_SIZE) + head_dim // 2)
            q_lower = tl.load(Q_row_start + tl.arange(0, BLOCK_SIZE) + head_dim // 2)
            q_cos_upper = tl.load(cos_row_start + tl.arange(0, BLOCK_SIZE))
            q_cos_lower = tl.load(cos_row_start + tl.arange(0, BLOCK_SIZE))
            q_sin_upper = tl.load(sin_row_start + tl.arange(0, BLOCK_SIZE))
            q_sin_lower = tl.load(sin_row_start + tl.arange(0, BLOCK_SIZE))

        # Compute q_{rot} = [q_{1}, q_{2}, ... q_{d_{head}/2}] * cos + [-q_{d_{head}/2 + 1}, ... -q_{d_{head}}] * sin
        # Compute q_{rot} = [q_{1}, q_{2}, ... q_{d_{head}/2}] * cos - [-q_{d_{head}/2 + 1}, ... -q_{d_{head}}] * sin
        q_upper = tl.where(
            tl.arange(0, BLOCK_SIZE) < head_dim // 2,
            q_upper * q_cos_upper - q_sin_upper * q_lower,
            q_upper,
        )
        q_sin = tl.where(
            tl.arange(0, BLOCK_SIZE) < head_dim // 2,
            q_sin_upper * q_lower + q_cos_upper * q_upper,
            q_sin_upper,
        )

        tl.store(Q_row_start + tl.arange(0, BLOCK_SIZE), q)
        tl.store(Q_row_start + tl.arange(0, BLOCK_SIZE) + head_dim // 2, q_sin)

    def _rope_embedding_forward_impl(Q, cos, sin):
        Q = Q.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        seqlen, n_heads, head_dim = Q.shape
        assert head_dim % 2 == 0
        head_dim_half = head_dim // 2
        n_heads_rope_group = 4
        BLOCK_SIZE, num_warps = calculate_settings(head_dim)
        n_groups = (n_heads + n_heads_rope_group - 1) // n_heads_rope_group
        Q_rope = torch.empty_like(Q)

        def grid(META):
            return (triton.cdiv(seqlen, META["BLOCK_SIZE"]), n_groups)

        _rope_embedding[grid](Q, Q.stride(0), cos, cos.stride(0), sin, sin.stride(0), seqlen, head_dim, n_heads,
                               False, BLOCK_SIZE, n_heads_rope_group)

        return Q_rope

    def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
        dY = dY.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        seqlen, n_heads, head_dim = dY.shape
        assert head_dim % 2 == 0
        head_dim_half = head_dim // 2

        dY_rope = torch.empty_like(dY)

        def grid(META):
            return (triton.cdiv(seqlen, META["BLOCK_SIZE"]), n_groups)

        _rope_embedding[grid](dY, dY.stride(0), cos, cos.stride(0), sin, sin.stride(0), seqlen, head_dim, n_heads,
                               True, BLOCK_SIZE, head_dim_half, num_warps=num_warps)

        return dY_rope
