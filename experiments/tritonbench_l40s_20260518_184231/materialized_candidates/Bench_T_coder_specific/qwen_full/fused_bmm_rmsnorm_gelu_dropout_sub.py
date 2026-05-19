import torch
import triton
import triton.language as tl
from triton.ops.layer_norm import rms_norm
from torch import Tensor
from torch.autograd.function import Function

class FusedBMmRmsNormGeluDropoutSub(Function):
    @staticmethod
    def forward(ctx, input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, out=None):
        if out is None:
            out = torch.empty_like(input1, dtype=input1.dtype)
        else:
            out.resize_as_(input1)

        if approximate == 'tanh':
            input1 = input1 * 0.7978845608028654  # alpha = 0.7978845608028654
            input2 = input2 * 0.7978845608028654  # alpha = 0.7978845608028654

        input1_ = input1
        input2_ = input2

        input1 = input1.contiguous()
        input2 = input2.contiguous()

        out_b, out_n, out_m = out.shape
        out_m_1, out_m_2 = input2.size(-2), input2.size(-1)
        out_n_1, out_n_2 = input1.size(-2), input1.size(-1)
        assert out_m_1 == out_n_2, "incompatible hidden size - expected hidden size to be {0} but got {1} ".format(out_m_1, out_n_2)
        assert out_m_2 == normalized_shape, "incompatible feature size - expected feature size to be {0} but got {1} ".format(out_m_2, normalized_shape)

        input1 = input1.view(out_b * out_n, out_m_1, out_m_2)
        input2 = input2.view(out_b * out_n, out_m_1, out_m_2)
        out = out.view(out_b * out_n, out_m_1, out_m_2)

        if dropout_p > 0.0:
            dropout_mask = torch.rand((out_b * out_n, out_m_1), dtype=torch.float32, device=input1.device) > dropout_p
            dropout_mask = (dropout_mask / (1.0 - dropout_p)).to(input1.dtype)
        else:
            dropout_mask = torch.ones((out_b * out_n, out_m_1), dtype=torch.float32, device=input1.device)

        for i in range(0, out_b * out_n, BLOCK_SIZE_N):
            i_n = i + tl.arange(0, BLOCK_SIZE_N)
            mask_n = i_n < out_b * out_n
            for j in range(0, out_m_1, BLOCK_SIZE_M):
                j_m = j + tl.arange(0, BLOCK_SIZE_M)
                mask_m = j_m < out_m_1
                for k in range(0, out_m_2, BLOCK_SIZE_K):
                    k_m = k + tl.arange(0, BLOCK_SIZE_K)
                    mask_k = k_m < out_m_2

                    mask = mask_n[:, None] & mask_m[None, :] & mask_k[None, :]

                    a = tl.load(input1_ + i_n[:, None, None] * out_m_1 * out_m_2 + j_m[None, :, None] * out_m_2 + k_m[None, None, :])
                    b = tl.load(input2_ + i_n[:, None, None] * out_m_1 * out_m_2 + j_m[None, :, None] * out_m_2 + k_m[None, None, :])

                    a = a * dropout_mask[i_n[:, None], j_m[None, :]]
                    c = tl.dot(a, b, allow_tf32=False)

                    d = c.to(tl.float32)
                    e = rms_norm(d, mask, eps, BLOCK_SIZE_K)
                    f = gelu_new(e)
                    g = f.to(tl.float32)

                    tl.store(out + i_n[:, None, None] * out_m_1 * out_m_2 + j_m[None, :, None] * out_m_2 + k_m[None, None, :], g)

        out = out + other
        out = out.view((out_b, out_n, out_m))
        return out

def fused_bmm_rmsnorm_gelu_dropout_sub(input1: Tensor, input2: Tensor, other: Tensor, normalized_shape: int, dropout_p: float = 0.5, training: bool = True, approximate: str = 'none', eps: float = 1e-5, *, out: Tensor = None) -> Tensor:
    if approximate != 'none' and approximate != 'tanh':
        raise ValueError(f"Invalid approximate value: {approximate}")

    if approximate == 'tanh':
        assert dropout_p == 0.0, "dropout_p must be 0.0 when approximate is 'tanh'"

    if approximate == 'none':
        return FusedBMmRmsNormGeluDropoutSub.apply(input1, input2, other, normalized_shape, dropout_p, training, approximate, eps, out)
    else:
        return FusedTanhGelu.apply(input1, input2, other, normalized_shape, dropout_p, training, approximate, eps, out)

@triton.jit
def gelu_new(x):
    pi = 3.141592653589793
    a = tl.math.sqrt(2.0 / pi)
    b = x + 0.044715 * x * x
    c = tl.math.tanh(a * b)
    d = 0.5 * x * (1.0 + c)
    return d

def gelu(input):
    return gelu_new(input)
