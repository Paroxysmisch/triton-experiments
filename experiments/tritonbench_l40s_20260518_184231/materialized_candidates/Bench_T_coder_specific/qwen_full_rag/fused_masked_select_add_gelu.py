import logging
import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, pow
from triton.language.math import tanh

@triton.jit
def fused_masked_select_add_gelu_kernel(X, M, O, alpha, approximate):
    # Convert inputs to float32 for better precision
    X_fp32 = X.to(tl.float32)
    M_bool = M.to(tl.int1)
    # Apply mask and compute GELU
    if approximate == "none":
        X_gelu = tl.where(M_bool, 0.5 * X_fp32 * (1 + erf(0.7071067811 * X_fp32)), 0)
    else:
        X_gelu = tl.where(
            M_bool,
            0.5
            * X_fp32
            * (1 + tanh(0.79788456 * X_fp32 * (1 + 0.044715 * pow(X_fp32, 2)))),
            0,
        )
    # Scale and add
    S = X_gelu * alpha + O
    return S

class FusedMaskedSelectAddGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C, D, e, approximate):
        # Log debug information
        logging.debug("FUSED MASKED SELECT ADD GELU")
        # Assert that inputs are contiguous
        assert A.is_contiguous()
        assert B.is_contiguous()
        # Call the Triton kernel
        return fused_masked_select_add_gelu_kernel(A, B, C, D, e, approximate)

def fused_masked_select_add_gelu(A, B, C, *, alpha=1, approximate="none", out=None):
    # Wrapper function for using FusedMaskedSelectAddGELU class
    if out is None:
        out = torch.empty_like(C)
    else:
        assert out.is_contiguous()
    # Apply the custom autograd function
    return FusedMaskedSelectAddGELU.apply(A, B, C, out, alpha, approximate)
