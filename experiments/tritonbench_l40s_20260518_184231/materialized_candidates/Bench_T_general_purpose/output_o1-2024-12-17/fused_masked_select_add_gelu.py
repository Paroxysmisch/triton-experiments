import torch
import triton
import triton.language as tl

# ---------------------------
# Triton kernel
# ---------------------------
@triton.jit
def _fused_masked_select_add_gelu_kernel(
    input_ptr,   # ptr float
    mask_ptr,    # ptr bool/int
    other_ptr,   # ptr float (same shape as input)
    cumsum_ptr,  # ptr int
    out_ptr,     # ptr float
    n_elems,     # int
    alpha,       # float
    approx_id,   # int (0 = 'none', 1 = 'tanh')
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elems

    # Load data
    x  = tl.load(input_ptr  + block_start, mask=mask, other=0.0)
    mk = tl.load(mask_ptr   + block_start, mask=mask, other=0)
    o  = tl.load(other_ptr  + block_start, mask=mask, other=0.0)
    cs = tl.load(cumsum_ptr + block_start, mask=mask, other=0)

    # Fused op:
    # if mk != 0,  idx_out = cs - 1
    # out[idx_out] = GELU( x + alpha*o )
    # else do nothing
    # We handle approx_id to choose the GELU formula:
    #   approx_id=0 -> exact   (0.5 * x * (1 + erf(x/sqrt(2))))
    #   approx_id=1 -> approx (0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3)) ))
    
    # compute S
    s = x + alpha * o
    
    # compute GELU
    if approx_id == 0:
        # exact
        inv_sqrt2 = 1.0 / tl.sqrt(2.0)
        erf_val = tl.erf(s * inv_sqrt2)
        gelu_val = 0.5 * s * (1.0 + erf_val)
    else:
        # tanh approximation
        # 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
        cst = tl.sqrt(2.0 / 3.141592653589793)
        inner = cst * (s + 0.044715 * s * s * s)
        gelu_val = 0.5 * s * (1.0 + tl.tanh(inner))

    # store results only where mk != 0
    mk_bool = mk.to(tl.int32)
    valid_offset = cs - 1  # zero-based index in output
    should_store = mk_bool != 0
    tl.store(out_ptr + valid_offset, gelu_val, mask=should_store)


# ---------------------------
# Python Wrapper
# ---------------------------
def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    """
    fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None) -> Tensor
    Math:
        Z = masked_select(X, M)
        S = Z + alpha * O
        Y = GELU(S)
    where
        - X, M, O are broadcastable / flattenable.
        - approximate can be 'none' or 'tanh' for faster approximate GELU.
        - The output is a 1-D tensor containing the GELU values of the selected elements.
    """
    # Flatten the input / mask to 1D
    X = input.contiguous().view(-1)
    M = mask.to(dtype=torch.bool, copy=True).contiguous().view(-1)
    # Broadcast 'other' to match X's shape if needed
    # (PyTorch can broadcast automatically, but replicate here for clarity)
    O = other.expand_as(input).contiguous().view(-1)

    # Convert mask to int for cumsum
    M_int = M.to(dtype=torch.int32)
    cumsum_mask = torch.cumsum(M_int, dim=0)
    if cumsum_mask.numel() == 0:
        # Edge case: empty input, just return an empty result
        if out is None:
            return torch.empty(0, dtype=input.dtype, device=input.device)
        else:
            return out.resize_(0)

    total_selected = cumsum_mask[-1].item()
    if out is None:
        out = torch.empty(total_selected, dtype=input.dtype, device=input.device)
    else:
        # Resize out appropriately
        out.resize_(total_selected)

    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: ((X.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    approx_id = 0 if approximate == 'none' else 1
    alpha_f32 = float(alpha)

    _fused_masked_select_add_gelu_kernel[grid](
        input_ptr=X, 
        mask_ptr=M_int,
        other_ptr=O,
        cumsum_ptr=cumsum_mask,
        out_ptr=out,
        n_elems=X.numel(),
        alpha=alpha_f32,
        approx_id=approx_id,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
