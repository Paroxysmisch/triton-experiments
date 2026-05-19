import torch
import triton
import triton.language as tl

@triton.jit
def normalize(x, mask, out, n_dim_reduced, n_elements, n_features, square_add):
    # Map the program id to the element of x it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, n_features)
    x_ptrs = x + row * n_features + cols
    mask_ptrs = mask + cols
    out_ptrs = out + row * n_features + cols
    x_values = tl.load(x_ptrs, mask=mask_ptrs, other=0.0)
    x_sq = x_values * x_values
    if square_add:
        square_add_val = tl.load(x_ptrs)
        x_sq += square_add_val * square_add_val
    sum_sq = tl.sum(x_sq, axis=0)
    rsqrt_norm = tl.math.rsqrt(sum_sq)
    x_norm = x_values * rsqrt_norm
    tl.store(out_ptrs, x_norm)

@triton.jit
def compute_loss(x, y, margin, n_elements, n_features, loss, square_add):
    # Map the program id to the element of x it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, n_features)
    x_ptrs = x + row * n_features + cols
    y_ptrs = y + row * n_features + cols
    x_values = tl.load(x_ptrs)
    y_values = tl.load(y_ptrs)
    dot = tl.sum(x_values * y_values, axis=0)
    if square_add:
        square_add_val = tl.load(y_ptrs)
        dot += square_add_val * square_add_val
    loss_value = tl.math.log(tl.math.exp(dot) + tl.math.exp(-dot - margin))
    tl.store(loss + row, -loss_value)

@triton.jit
def reduce_loss(loss, n_elements, reduction, out):
    # Map the program id to the element of x it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, n_elements)
    loss_ptrs = loss + cols
    if reduction == "mean":
        loss_values = tl.load(loss_ptrs, mask=cols < n_elements, other=0.0)
        mean = tl.sum(loss_values, axis=0) / n_elements
        tl.store(out + row, mean)
    elif reduction == "sum":
        loss_values = tl.load(loss_ptrs, mask=cols < n_elements, other=0.0)
        sum = tl.sum(loss_values, axis=0)
        tl.store(out + row, sum)
    else:
        tl.store(out + row, 0.0)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor,
    input2: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0,
    reduction: str = 'mean',
) -> torch.Tensor:
    n_elements, n_features = input1.shape
    out1 = torch.empty((n_elements, n_features), dtype=torch.float32, device=input1.device)
    out2 = torch.empty((n_elements, n_features), dtype=torch.float32, device=input1.device)
    mask = torch.not_equal(input1, 0)
    n_dim_reduced = torch.tensor(1, dtype=torch.int32, device=input1.device)
    normalize[n_elements,](
        input1,
        mask,
        out1,
        n_dim_reduced,
        n_elements,
        n_features,
        square_add=torch.tensor(0.0, dtype=torch.float32, device=input1.device),
    )
    normalize[n_elements,](
        input2,
        mask,
        out2,
        n_dim_reduced,
        n_elements,
        n_features,
        square_add=torch.tensor(0.0, dtype=torch.float32, device=input1.device),
    )
    loss = torch.empty((n_elements,), dtype=torch.float32, device=input1.device)
    compute_loss[n_elements,](
        out1,
        out2,
        torch.tensor(margin, dtype=torch.float32, device=input1.device),
        n_elements,
        n_features,
        loss,
        square_add=torch.tensor(0.0, dtype=torch.float32, device=input1.device),
    )
    loss_out = torch.empty((1,), dtype=torch.float32, device=input1.device)
    reduce_loss[1,](
        loss,
        n_elements,
        reduction,
        loss_out,
    )
    return loss_out
