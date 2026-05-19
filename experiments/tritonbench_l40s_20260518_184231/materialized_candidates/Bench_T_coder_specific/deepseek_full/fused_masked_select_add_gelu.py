import torch
import triton
import triton.language as tl
from flag_gems.utils.gelu_utils import gelu, gelu_approx

@triton.jit
def fused_masked_select_add_gelu_(
    input,
    mask,
    other,
    alpha,
    approximate,
    **meta,
):
    idx = tl.program_id(axis=0)
    input_ = tl.load(input + idx)
    mask_ = tl.load(mask + idx)
    other_ = tl.load(other + idx)
    if approximate == "tanh":
        gelu_ = gelu_approx(input_, other_, alpha, meta["N"])
    else:
        gelu_ = gelu(input_, other_, alpha, meta["N"])
    tl.store(input + idx, gelu_)

def fused_masked_select_add_gelu(
    input: torch.Tensor,
    mask: torch.Tensor,
    other: torch.Tensor,
    *,
    alpha: Union[float, torch.Tensor] = 1,
    approximate: str = "none",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, device=input.device, dtype=input.dtype)
    input_ = input if out is None else out
    assert (
        input_.shape == mask.shape == other.shape
    ), "shape of input, mask and other must be the same"
    assert approximate in {
        "none",
        "tanh",
    }, f"approximate must be 'none' or 'tanh', got {approximate}"
    N = input_.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(input_.device.index):
        fused_masked_select_add_gelu_[grid](
            input_,
            mask,
            other,
            alpha,
            approximate,
            N=N,
        )
    return input_
