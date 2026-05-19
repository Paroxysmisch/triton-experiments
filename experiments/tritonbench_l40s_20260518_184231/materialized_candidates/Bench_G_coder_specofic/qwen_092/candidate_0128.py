import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X: tl.tensor, 
    OUT: tl.tensor, 
    COS: tl.tensor, 
    SIN: tl.tensor, 
    CU_SEQLENS: tl.tensor,
    pid_batch: tl.index,
    pid_head: tl.index,
    pid_m: tl.index,
    in_interleaved: tl.constexpr,
    out_interleaved: tl.constexpr,
    in_place: tl.constexpr,
    conjugate: tl.constexpr,
    batch_stride: tl.constexpr,
    head_stride: tl.constexpr,
    seq_stride: tl.constexpr,
    seq_len: tl.constexpr,
    max_seq_len: tl.constexpr,
    block_size: tl.constexpr,
    block_size_y: tl.constexpr,
    block_size_m: tl.constexpr,
    num_heads: tl.constexpr,
    num_batches: tl.constexpr,
):
    """
    Triton kernel for performing rotary position encoding on a tensor X.
    """
    # Calculate indices
    b = pid_batch * batch_stride
    h = pid_head * head_stride
    m = pid_m * seq_stride
    
    # Load data from X
    if in_interleaved:
        x0 = tl.load(X + (b + h + m) * 2)
        x1 = tl.load(X + (b + h + m) * 2 + 1)
    else:
        x0 = tl.load(X + (b + h + m))
    
    # Load cosine and sine values
    cos = tl.load(COS + (h + m % max_seq_len) * 2)
    sin = tl.load(SIN + (h + m % max_seq_len) * 2)
    
    # Apply rotary transformation
    if conjugate:
        if in_interleaved:
            y0 = x1 * cos - x0 * sin
            y1 = x0 * cos + x1 * sin
        else:
            y0 = x0 * cos + x1 * sin
            y1 = x1 * cos - x0 * sin
    else:
        if in_interleaved:
            y0 = x0 * cos - x1 * sin
            y1 = x1 * cos + x0 * sin
        else:
            y0 = x0 * cos + x1 * sin
            y1 = x1 * cos - x0 * sin
    
    # Store result in OUT
    if out_interleaved:
        if in_place:
            tl.store(X + (b + h + m) * 2, y0)
            tl.store(X + (b + h + m) * 2 + 1, y1)
        else:
            tl.store(OUT + (b + h + m) * 2, y0)
            tl.store(OUT + (b + h + m) * 2 + 1, y1)
    else:
        if in_place:
            tl.store(X + (b + h + m), y0)
        else:
            tl.store(OUT + (b + h + m), y0)
