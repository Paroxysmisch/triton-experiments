import torch
import triton
import triton.language as tl

def get_triton_grid(M, BLOCK_M: int):
    grid_m = triton.cdiv(M, BLOCK_M)
    grid = (grid_m,)
    return grid

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1}),
        triton.Config({"BLOCK_M": 2}),
        triton.Config({"BLOCK_M": 4}),
        triton.Config({"BLOCK_M": 8}),
        triton.Config({"BLOCK_M": 16}),
        triton.Config({"BLOCK_M": 32}),
        triton.Config({"BLOCK_M": 64}),
    ],
    key=["output_size"],
)
@triton.jit
def fused_repeat_interleave_log_softmax_kernel(
    output_ptr,
    input_ptr,
    repeats_ptr,
    scale_ptr,
    sequence_length,
    output_size,
    max_sequence_length,
    BLOCK_M: tl.constexpr,
):
    program_idx = tl.program_id(axis=0)
    repeat_idx = tl.program_id(axis=1)
    start_token_idx = program_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    token_idx = start_token_idx + tl.arange(0, BLOCK_M)
    valid_mask = token_idx < max_sequence_length
    token_idx_clamp = tl.minimum(token_idx, max_sequence_length - 1)
    output_idx = token_idx_clamp * output_size + repeat_idx
    output_ptr_masked = output_ptr + output_idx
    input_ptr_base = input_ptr + token_idx * output_size
    repeats_ptr_base = repeats_ptr + token_idx
    scale_ptr_base = scale_ptr + token_idx

    for _ in range(BLOCK_M):
        token_idx_val = token_idx
        token_idx_val = tl.multiple_of(token_idx_val, BLOCK_M)
        curr_repeats = tl.load(repeats_ptr_base + token_idx_val, mask=valid_mask, other=0)
        curr_scale = tl.load(scale_ptr_base + token_idx_val, mask=valid_mask, other=0)
        prev_output = tl.zeros((BLOCK_M,), tl.float32) - float("inf")
        curr_input_ptr = input_ptr_base + token_idx_val * output_size
        curr_output_ptr = output_ptr_masked + token_idx_val * output_size
        for _ in range(curr_repeats):
            __input = tl.load(curr_input_ptr, mask=valid_mask, other=float("-inf"))
            curr_max = tl.maximum(prev_output, __input + curr_scale)
            temp1 = tl.exp(prev_output - curr_max)
            temp2 = tl.exp(__input + curr_scale - curr_max)
            prev_output = curr_max + tl.log(temp1 + temp2)
            curr_input_ptr += output_size
        tl.store(curr_output_ptr, prev_output, mask=valid_mask)


class FusedRepeatInterleaveLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, repeats, dim=None, scale=None, output_size=None):
        if dim is None:
            input = input.flatten()
            dim = 0
        else:
            input = input.contiguous()

        scale = (
            torch.ones_like(repeats, dtype=torch.float32, device=repeats.device)
            if scale is None
            else scale.contiguous()
        )
        assert input.dim() == 2, "only accept 2D tensor now"
        batch, seq_len = input.shape
        assert repeats.shape == (batch, seq_len)
        assert scale.shape == (batch, seq_len)
        max_seq_len = int(max(list(repeats.sum(dim=1))))
        output_size = repeats.max().to(torch.int32)

        ctx.save_for_backward(input, repeats, scale)
        ctx.dim = dim
        ctx.scale = scale
        ctx.output_size = output_size
        ctx.sequence_length = seq_len

        # prepare output
        output = torch.empty(batch, seq_len, output_size, device=input.device, dtype=input.dtype)

        grid = get_triton_grid(seq_len, BLOCK_M=32)

        fused_repeat_interleave_log_softmax_kernel[grid](
            output,
            input,
            repeats,
            scale,
            seq_len,
            output_size,
            max_seq_len,
            BLOCK_M=32,
        )

        return output


def fused_repeat_interleave_log_softmax(input, repeats, dim=None, scale=None, output_size=None):
    assert input.is_contiguous()
    return FusedRepeatInterleaveLogSoftmax.apply(input, repeats, dim, scale, output_size)
