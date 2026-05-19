import torch
from torch.autograd import Function
import triton
import triton.language as tl

class LinearLayer(Function):
    @staticmethod
    @triton.jit
    def forward(ctx, A, B, C, bias=None, activation=0, fast_gelu_alpha=0.0):
        M, K = A.shape
        K, N = B.shape
        if C is not None:
            assert C.shape == (M, N), "Output tensor shape mismatch"

        if bias is not None:
            assert bias.shape == (N,), "Bias tensor shape mismatch"

        output = torch.empty((M, N), device=A.device, dtype=A.dtype)
        A_row_stride = A.stride(0)
        A_col_stride = A.stride(1)
        B_row_stride = B.stride(0)
        B_col_stride = B.stride(1)
        C_row_stride = C.stride(0) if C is not None else 0
        C_col_stride = C.stride(1) if C is not None else 0
        bias_row_stride = bias.stride(0) if bias is not None else 0
        bias_col_stride = bias.stride(1) if bias is not None else 0
        output_row_stride = output.stride(0)
        output_col_stride = output.stride(1)

        kernel_fma[triton.cdiv(M, 32), triton.cdiv(N, 32)](
            A, A_row_stride, A_col_stride,
            B, B_row_stride, B_col_stride,
            C, C_row_stride, C_col_stride,
            bias, bias_row_stride, bias_col_stride,
            output, output_row_stride, output_col_stride,
            M, N, K,
            1.0, 0.0,
            activation,
            fast_gelu_alpha
        )

        ctx.save_for_backward(A, B, bias, output)
        return output

    @staticmethod
    @triton.jit
    def backward(ctx, grad_output):
        A, B, bias, output = ctx.saved_tensors
        M, K = A.shape
        K, N = B.shape

        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        A_row_stride = A.stride(0)
        A_col_stride = A.stride(1)
        B_row_stride = B.stride(0)
        B_col_stride = B.stride(1)
        grad_output_row_stride = grad_output.stride(0)
        grad_output_col_stride = grad_output.stride(1)

        kernel_fma[triton.cdiv(M, 32), triton.cdiv(N, 32)](
            grad_output, grad_output_row_stride, grad_output_col_stride,
            B, B_row_stride, B_col_stride,
            grad_A, A_row_stride, A_col_stride,
            None, 0, 0,
            None, 0, 0,
            M, N, K,
            1.0, 0.0,
            0, 0.0
        )

        kernel_fma[triton.cdiv(M, 32), triton.cdiv(N, 32)](
            A, A_row_stride, A_col_stride,
            grad_output, grad_output_row_stride, grad_output_col_stride,
            grad_B, B_row_stride, B_col_stride,
            None, 0, 0,
            None, 0, 0,
            M, N, K,
            1.0, 0.0,
            0, 0.0
        )

        if bias is not None:
            kernel_fma[triton.cdiv(M, 32), triton.cdiv(N, 32)](
                grad_output, grad_output_row_stride, grad_output_col_stride,
                None, 0, 0,
                grad_bias, 0, 1,
                None, 0, 0,
                M, N, K,
                1.0, 0.0,
                0, 0.0
            )

        return grad_A, grad_B, grad_output, grad_bias, None, None

def linear_layer(input, weight, bias=None, activation=0, fast_gelu_alpha=0.0):
    return LinearLayer.apply(input, weight, bias, activation, fast_gelu_alpha)
