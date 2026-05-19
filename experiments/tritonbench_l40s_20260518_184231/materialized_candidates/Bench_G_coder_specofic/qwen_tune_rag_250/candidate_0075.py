# Function implementation to run the sampled addmm kernel

def sampled_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[Tuple[int, int, int]] = None
):
    # Function implementation for sampled addmm operation

def bsr_softmax(
    input: torch.Tensor,
    crow_indices: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    max_grid: Optional[Tuple[int, int, int]] = None
):
    # Function implementation for BSR softmax operation
