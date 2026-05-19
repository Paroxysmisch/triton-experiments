import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_logsoftmax_dropout(
    input,
    vec,
    p: tl.constexpr,
    training: tl.constexpr,
    inplace: tl.constexpr,
    dim: tl.constexpr,
    out=None,
):
    # Define the matrix-vector multiplication
    z = tl.dot(input, vec)

    # Compute the log-softmax
    if dim == 0:
        z_max = tl.max(z, 0)
        s = z - z_max - tl.log(tl.sum(tl.exp(z - z_max), 0))
    else:
        z_max = tl.max(z, 1)
        s = z - z_max - tl.log(tl.sum(tl.exp(z - z_max), 1))

    # Apply dropout
    if training:
        s = tl.where(tl.rand(tl.shape(s)) > p, s / (1 - p), 0.0)

    if inplace:
        if out is None:
            return z
        else:
            return s
    else:
        return s

def test_fused_mv_logsoftmax_dropout():
    # Test the function with random inputs
    input = torch.randn(2, 3, device="cuda")
    vec = torch.randn(3, device="cuda")
    out = torch.empty(2, device="cuda")
    p = 0.5
    training = True
    inplace = False

    # Call the function
    fused_mv_logsoftmax_dropout(input, vec, p, training, inplace, 0, out=out)
    assert out.allclose(torch.matmul(input, vec))

    # Test with inplace=True
    result = fused_mv_logsoftmax_dropout(input, vec, p, training, True, 0)
    assert result.allclose(torch.matmul(input, vec))

    # Test with training=False
    result = fused_mv_logsoftmax_dropout(input, vec, p, False, False, 0)
    assert result.allclose(torch.matmul(input, vec))
