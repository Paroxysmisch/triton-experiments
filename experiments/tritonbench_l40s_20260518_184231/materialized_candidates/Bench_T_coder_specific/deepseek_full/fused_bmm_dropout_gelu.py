import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from ..utils.gelu import gelu_wrapper
from ..utils.misc import calculate_settings
from ..utils.model_size import model_size

class FusedBmmDropoutGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, p, training, inplace, out=None, bf16=False, approximate="none"):
        if not training:
            p = 0.0
        output = torch.empty_like(x) if out is None else out

        # Settings
        BLOCKSIZE, numSM = calculate_settings(x)

        # Grid
        grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCKSIZE_M"]), x.shape[1], y.shape[2])

        # Kernel
        fused_bmm_dropout_gelu[grid](x, y, output, p, training=training, inplace=inplace, BLOCKSIZE_M=BLOCKSIZE,
                                     NUM_SM=numSM,
                                     GROUP_M=min(512 // BLOCKSIZE, 8),
                                     eval_stages=model_size() < 3000,
                                     eval_stages_final=model_size() < 6000,
                                     num_warps=4 if bf16 else 16, bf16=bf16,
                                     early_config_prune=(
                                         early_config_prune if bf16 else None),
                                     estimate_time=(None if bf16 else estimate_matmul_time),
                                     save_bf16_config=(model_size() >= 6000),
                                     allow_tf32=(model_size() < 6000),
                                     approximate=approximate)

        ctx.save_for_backward(x, y)
        ctx.p = p
        ctx.inplace = inplace
        ctx.training = training

        return output

    @staticmethod
    def backward(ctx, output_grad):
        # Retrieve saved tensors
        (x, y) = ctx.saved_tensors

        # Settings
        BLOCKSIZE, numSM = calculate_settings(x)

        # Autograd check
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None, None, None, None

        # Grad w.r.t. input1
        input1_grad = torch.empty_like(x) if ctx.needs_input_grad[0] else None

        # Grad w.r.t. input2
        input2_grad = torch.empty_like(y) if ctx.needs_input_grad[1] else None

        # Grid
        grid = lambda meta: (triton.cdiv(x.shape[0], meta["BLOCKSIZE_M"]), x.shape[1], y.shape[2])

        # Kernel
        fused_bmm_dropout_gelu_bw1_bw2[grid](x, y, output_grad, input1_grad, input2_grad, ctx.p,
                                             ctx.training, ctx.inplace,
                                             BLOCKSIZE_M=BLOCKSIZE, NUM_SM=numSM,
                                             GROUP_M=min(512 // BLOCKSIZE, 8), num_warps=16)

        return (input1_grad, input2_grad, None, None, None, None)

@triton.jit
def fused_bmm_dropout_gelu(x_ptr, y_ptr, output_ptr, p, training: tl.constexpr, inplace: tl.constexpr,
                           BLOCKSIZE_M: tl.constexpr, NUM_SM: tl.constexpr, GROUP_M: tl.constexpr,
                           eval_stages: tl.constexpr, eval_stages_final: tl.constexpr, num_warps: tl.constexpr,
                           bf16: tl.constexpr, early_config_prune, estimate_time, save_bf16_config,
                           allow_tf32,
                           approximate: tl.constexpr):
    # Kernel code...

@triton.jit
def fused_bmm_dropout_gelu_bw1_bw2(x_ptr, y_ptr, out_grad_ptr, input1_grad_ptr, input2_grad_ptr, p,
                                   training: tl.constexpr, inplace: tl.constexpr,
                                   BLOCKSIZE_M: tl.constexpr, NUM_SM: tl.constexpr, GROUP_M: tl.constexpr,
                                   num_warps: tl.constexpr):
    # Kernel code...

def fused_bmm_dropout_gelu(input1: Tensor, input2: Tensor, p: float = 0.5, training: bool = True,
                           inplace: bool = False,
                           out: Optional[Tensor] = None, bf16: bool = False, approximate: str = "none"):
    return FusedBmmDropoutGELU.apply(input1, input2, p, training, inplace, out, bf16, approximate)
