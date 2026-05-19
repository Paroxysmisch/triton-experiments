import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    lora_indices,
    input_ptr,
    lora_ptr,
    out_ptr,
    scaling,
    stride_lora_batch,
    stride_lora_n,
    stride_lora_k,
    stride_input_batch,
    stride_input_n,
    stride_input_k,
    stride_out_batch,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    lora_index = tl.load(lora_indices + pid_batch)
    if lora_index == -1:
        return
    batch_offsets = pid_batch * stride_lora_batch + lora_index * stride_lora_batch
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    input_ptr = input_ptr + batch_offsets * stride_input_batch
    lora_ptr = lora_ptr + batch_offsets * stride_lora_batch
    out_ptr = out_ptr + batch_offsets * stride_out_batch

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, BLOCK_K, BLOCK_K // SPLIT_K):
        k_offsets = k + tl.arange(0, BLOCK_K // SPLIT_K)
        a_ptr = tl.make_block_ptr(
            base=input_ptr,
            shape=(BLOCK_M, BLOCK_N),
            strides=(stride_input_k, stride_input_n),
            offsets=(k_offsets, n_offsets),
        )
        b_ptr = tl.make_block_ptr(
            base=lora_ptr,
            shape=(BLOCK_M, BLOCK_K),
            strides=(stride_lora_n, stride_lora_k),
            offsets=(n_offsets, k_offsets),
        )
        accumulator += tl.load(a_ptr) * tl.load(b_ptr)

    accumulator = accumulator.to(tl.float16)
    m, n = tl.meshgrid(
        tl.arange(0, BLOCK_M),
        tl.arange(0, BLOCK_N),
        indexing="ij",
    )
    out_ptrs = out_ptr + m * stride_out_n + n
    tl.atomic_add(out_ptrs, accumulator[m, n] * scaling)


def _bgmv_shrink(lora_indices, input, lora_weight, output, scaling):
    assert input.is_contiguous()
    assert lora_weight.is_contiguous()
    assert output.is_contiguous()

    _, lora_batch, _ = lora_weight.shape
    _, in_n, in_k = input.shape
    _, lora_n, lora_k = lora_weight.shape
    _, out_n = output.shape

    BLOCK_N = triton.next_power_of_2(in_n)
    BLOCK_K = triton.next_power_of_2(max(in_k, lora_k))
    grid = (lora_batch, 1)

    num_warps = 4
    if BLOCK_K >= 4096:
        num_warps = 8
    if BLOCK_K >= 8192:
        num_warps = 16

    _bgmv_shrink_kernel[grid](
        lora_indices,
        input,
        lora_weight,
        output,
        scaling,
        lora_weight.stride(0),
        lora_weight.stride(1),
        lora_weight.stride(2),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=in_n,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=1,
        IS_TRITON_22=triton.__version__ >= "2.2.0",
        num_warps=num_warps,
    )


class _BGEMVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_indices, lora_weight, output_device=None):
        scaling = 1.0

        assert lora_weight.shape[0] == input.shape[0]
        assert lora_weight.shape[1] == input.shape[1] or lora_weight.shape[1] == 1
        assert lora_weight.shape[2] == input.shape[2] or lora_weight.shape[2] == 1

        input_ = input.contiguous()
        lora_weight_ = lora_weight.contiguous()

        output_device_ = torch.device(
            "cuda"
            if not output_device
            else (
                "cuda" if output_device.startswith("cuda") else "cpu"
            )  # make sure it is cuda or cpu
        )
        output = torch.zeros(
            input.shape[0], input.shape[1], device=output_device_
        )  # type: ignore

        lora_weight_ = lora_weight_.to(torch.float16)
        input_ = input_.to(torch.float16)
        output = output.to(torch.float16)

        _bgmv_shrink(
            lora_indices,
            input_,
            lora_weight_,
            output,
            scaling,
        )

        ctx.save_for_backward(input, lora_weight, lora_indices)
        ctx.scaling = scaling
        ctx.output_device = output_device_

        return output


def bgemv(
    input: torch.Tensor,
    lora_indices: torch.Tensor,
    lora_weight: torch.Tensor,
    output_device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Applies Low-Rank Adaptation (LORA) to the input.
    Args:
        input: input tensor of shape (batch, seq_len, d_model)
        lora_indices: tensor of shape (batch,) containing the indices of the LORA
            matrices to apply. An index of -1 means no LORA should be applied.
        lora_weight: tensor of shape (n_batches, d_model, d_model_lora)
            or (1, d_model, d_model_lora) containing the LORA matrices.
        output_device: device on which to output the result. Defaults to the
            device of the input.
    Returns:
        The result of applying LORA to the input.
    """
    return _BGEMVFunction.apply(input, lora_indices, lora_weight, output_device)
