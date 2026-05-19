import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    M, N, stride_am, stride_ak, stride_v, stride_vk, stride_om, stride_ok,
    alpha, other_is_scalar: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_v = tl.arange(0, BLOCK_N)
    a_ptrs = input_ptr + (offs_am[:, None] * stride_am + offs_bn[None, :] * stride_ak)
    v_ptrs = vec_ptr + offs_v * stride_v

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, N, BLOCK_N):
        a = tl.load(a_ptrs, mask=offs_bn[None, :] < N - _, other=0.0)
        v = tl.load(v_ptrs, mask=offs_v < N - _, other=0.0)
        acc += a * v
        a_ptrs += BLOCK_N * stride_ak
        v_ptrs += BLOCK_N * stride_v

    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ok = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_ok * offs_ok[None, :]
    mask = (offs_om[:, None] < M) & (offs_ok[None, :] < N)

    if other_is_scalar:
        scalar = other_ptr
        sigmoid = tl.sigmoid(acc)
        # out = sigmoid - alpha * scalar
        out = tl.where(mask, sigmoid - alpha * scalar, 0.0)
    else:
        b_ptrs = other_ptr + stride_om * offs_om[:, None] + stride_ok * offs_ok[None, :]
        scalar = tl.load(b_ptrs, mask=mask, other=0.0)
        sigmoid = tl.sigmoid(acc)
        # out = sigmoid - alpha * scalar
        out = tl.where(mask, sigmoid - alpha * scalar, 0.0)

    tl.store(out_ptrs, out, mask=mask)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    r"""
    Fused operation combining matrix-vector multiplication, sigmoid activation, and subtraction.

    Args:
        input (Tensor) : input matrix A of shape `(n, m)`
        vec (Tensor) : input vector :math:`\mathbf{v}` of shape `(m)`
        other (Tensor or Number) : tensor or scalar :math:`b` to subtract from sigmoid output, scaled by :math:`\alpha`
        alpha (Number, optional) : scalar multiplier for ``other``. Default: ``1``
        out (Tensor, optional) : output tensor. Ignored if ``None``. Default: ``None``

    Returns:
        Tensor : output tensor

    .. note::
        - The shapes of ``input`` and ``vec`` must be broadcastable.
        - If ``other`` is a tensor, it must be broadcastable with the sigmoid result tensor.
        - Autograd is supported for gradient computation.
    """
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert vec.dim() == 1, "Vector must be a 1D tensor"

    M, N = input.shape
    vec_shape_broadcasted = vec.shape[:-1] + (1,) * (input.dim() - vec.dim())
    assert broadcastable_to(
        vec_shape_broadcasted, (N,)
    ), "The shapes of input and vec must be broadcastable"

    if out is None:
        out = torch.empty((M,), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (M,)
        assert out.dtype == input.dtype
        assert out.device == input.device

    other_is_scalar = isinstance(other, (int, float))
    if other_is_scalar:
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        assert other.dim() == 0 or other.dim() == 2, "Other must be a 2D tensor or scalar"
        if other.dim() == 2:
            assert other.shape == input.shape, "The shapes of input and other must be the same"
        else:
            other = torch.reshape(other, (1,))

    assert broadcastable_to(
        other.shape, (M,)
    ), "If other is a tensor, it must be broadcastable with the sigmoid result tensor"

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        M, N, input.stride(0), input.stride(1), vec.stride(0), vec.stride(1), out.stride(0), out.stride(1),
        alpha, other_is_scalar,
    )
    return out
