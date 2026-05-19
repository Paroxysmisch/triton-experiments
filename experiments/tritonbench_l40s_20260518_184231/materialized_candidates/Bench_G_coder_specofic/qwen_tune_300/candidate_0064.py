import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,  # shape: [b, h, n, k]
    lora_ptr,  # shape: [lora_index, h, n, k]
    out_ptr,  # shape: [b, h, n, k]
    lora_indices,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    scale: tl.constexpr,
    add_inputs: tl.constexpr,
    cast_type: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # split n dimension to many blocks
    # more registers in this case
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    # compute block ptr
    block_n_offset = pid_n * BLOCK_N
    input_block_ptr = (
        input_ptr
        + pid_b * h * n * k
        + pid_h * n * k
        + block_n_offset * k
        + tl.arange(0, BLOCK_K)
    )
    lora_block_ptr = (
        lora_ptr
        + lora_indices * h * n * k
        + pid_h * n * k
        + block_n_offset * k
        + tl.arange(0, BLOCK_K)
    )
    out_block_ptr = (
        out_ptr
        + pid_b * h * n * k
        + pid_h * n * k
        + block_n_offset * k
        + tl.arange(0, BLOCK_K)
    )

    if cast_type:
        input_block_ptr = input_block_ptr.to(tl.float32)
        lora_block_ptr = lora_block_ptr.to(tl.float32)

    if EVEN_K:
        input = tl.load(input_block_ptr)
        lora = tl.load(lora_block_ptr)
    else:
        input = tl.load(input_block_ptr, mask=tl.arange(0, BLOCK_K) < k, other=0.0)
        lora = tl.load(lora_block_ptr, mask=tl.arange(0, BLOCK_K) < k, other=0.0)

    # range [0, n)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # range [0, b)
    b_range = pid_b + tl.zeros((BLOCK_N, ), dtype=tl.int32)
    lora_index = tl.load(lora_indices + b_range, mask=n_range < n, other=0)
    # compute in this case is very slow
    # input = tl.where(n_range < n, input, 0.0)
    # lora = tl.where(n_range < n, lora, 0.0)

    # [BLOCK_N, BLOCK_K] * [BLOCK_K] -> [BLOCK_N]
    if add_inputs:
        output = tl.sum(input * lora, axis=1) * scale
        output = tl.where(n_range < n, output, 0.0)
        tl.store(out_block_ptr, output, mask=n_range < n)
    else:
        output = tl.sum(input * lora, axis=1) * scale
        tl.store(out_block_ptr, output, mask=n_range < n)


@torch.inference_mode()
def _bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor | None = None,
    lora_indices_tensor: torch.Tensor | None = None,
    scaling: float = 1.0,
    add_inputs: bool = False,
) -> torch.Tensor:
    """
    Args:
        inputs (torch.Tensor): shape must be [b, h, n, k]
        lora_a_weights (torch.Tensor): shape must be [lora_index, h, n, k]
        output_tensor (torch.Tensor | None): shape must be [b, h, n, k]. If it is None, output will be allocated in this function.
        lora_indices_tensor (torch.Tensor | None): shape must be [b]. It contains lora index for each batch element. If it is None, all batch elements will use lora index 0.
        scaling (float): Optional scaling factor.
        add_inputs (bool): If true, input will be scaled and added to output.
    Return:
        torch.Tensor: If output_tensor is None, output tensor will be returned.
    """
    # init
    b, h, n, k = inputs.shape
    lora_n, lora_h, lora_c, lora_k = lora_a_weights.shape
    assert lora_h == h
    assert lora_k == k
    if lora_indices_tensor is None:
        lora_indices_tensor = torch.zeros((b, ), dtype=torch.int32, device=inputs.device)
    else:
        assert lora_indices_tensor.shape == (b, )
    assert inputs.dtype == lora_a_weights.dtype

    # allocate output
    if output_tensor is None:
        output_tensor = torch.empty_like(inputs)

    # check
    assert output_tensor.shape == inputs.shape
    assert lora_a_weights.shape[1:] == (n, k)

    # run kernel
    grid = lambda META: (b, h, triton.cdiv(n, META["BLOCK_N"]))  # noqa

    # split n dimension to many blocks
    # more registers in this case
    _bgmv_expand_slice_kernel[grid](
        inputs,
        lora_a_weights,
        output_tensor,
        lora_indices_tensor,
        b,
        h,
        n,
        k,
        scale=scaling,
        add_inputs=add_inputs,
        cast_type=inputs.dtype == torch.float16,
        BLOCK_N=triton.next_power_of_2(n),
        BLOCK_K=triton.next_power_of_2(k),
        SPLIT_N=1,
        EVEN_K=k % 16 == 0,
    )
    return output_tensor
