import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def fused_recurrent_gated_abc_fwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    o,
    h0,
    ht,
    s_k_h,
    s_v_h,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
):
    # Kernel implementation...
    # (Code omitted for brevity)

@triton.jit
def fused_recurrent_gated_abc_bwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    do,
    dq,
    dk,
    dv,
    h0,
    s_k_h,
    s_v_h,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
):
    # Kernel implementation...
    # (Code omitted for brevity)

class FusedRecurrentGatedABCFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, s, g, scale=None, initial_state=None, output_final_state=False, reverse=False):
        # Forward pass implementation...
        # (Code omitted for brevity)

    @staticmethod
    @custom_bwd
    def backward(ctx, do, dht=None):
        # Backward pass implementation...
        # (Code omitted for brevity)

def fused_recurrent_gated_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Function implementation...
    # (Code omitted for brevity)
