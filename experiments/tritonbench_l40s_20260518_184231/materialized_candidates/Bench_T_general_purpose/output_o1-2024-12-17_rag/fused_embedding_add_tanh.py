import math
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_embedding_forward_kernel(
    out_ptr,
    idx_ptr,
    weight_ptr,
    num_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_ptr += pid
    out_ptr += pid * num_cols

    mask = tl.arange(0, BLOCK_SIZE) < num_cols
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(idx_ptr).to(tl.int32)
    weight_ptr += row_idx * num_cols
    emb = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + cols, emb, mask=mask)


@triton.jit(do_not_specialize=["padding_idx"])
def _fused_embedding_backward_kernel(
    grad_weight_ptr,
    grad_out_ptr,
    idx_ptr,
    padding_idx,
    HAS_PADDING_IDX: tl.constexpr,
    num_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_ptr += pid
    grad_out_ptr += pid * num_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < num_cols

    row_idx = tl.load(idx_ptr).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_weight_ptr += row_idx * num_cols
        gout = tl.load(grad_out_ptr + cols, mask=mask, other=0.0)
        tl.atomic_add(grad_weight_ptr + cols, gout, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_weight_ptr += row_idx * num_cols
            gout = tl.load(grad_out_ptr + cols, mask=mask, other=0.0)
            tl.atomic_add(grad_weight_ptr + cols, gout, mask=mask)


@triton.jit
def _indice_freq_kernel(
    freq_ptr,
    idx_ptr,
    total_elems: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE

    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems
    idx_val = tl.load(idx_ptr + offsets, mask=mask)
    tl.atomic_add(freq_ptr + idx_val, 1, mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def _grad_scale_kernel(
    grad_weight_ptr,
    freq_ptr,
    n_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in range(row_start, n_rows, row_step):
        scale_val = 1.0
        f = tl.load(freq_ptr + row_idx)
        if f > 1:
            scale_val = 1.0 / f
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < num_cols
        g = tl.load(grad_weight_ptr + row_idx * num_cols + cols, mask=mask)
        tl.store(grad_weight_ptr + row_idx * num_cols + cols, g * scale_val, mask=mask)


class _FusedEmbeddingAddTanhFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_indices,
        weight,
        other,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    ):
        # If max_norm is specified, renorm the weight
        if max_norm is not None:
            with torch.no_grad():
                norms = weight.norm(p=norm_type, dim=1, keepdim=True)
                exceeds = norms > max_norm
                scale = max_norm / norms.clamp_min(1e-7)
                weight[exceeds] = weight[exceeds] * scale[exceeds]

        indices = input_indices.contiguous()
        w = weight.contiguous()
        M = indices.numel()
        N = w.shape[-1]

        # Embedding forward via Triton
        BLOCK_SIZE = triton.next_power_of_2(N)
        out_shape = (*indices.shape, N)
        emb_out = torch.empty(out_shape, device=w.device, dtype=w.dtype)

        grid = lambda meta: (M,)
        _fused_embedding_forward_kernel[grid](
            emb_out,
            indices,
            w,
            N,
            BLOCK_SIZE,
        )

        # E + O
        S = emb_out + other

        # Tanh(S)
        Y = torch.tanh(S)

        ctx.save_for_backward(Y, indices)
        ctx.weight_shape = w.shape
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        ctx.num_indices = M
        ctx.num_cols = N

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y, indices = ctx.saved_tensors
        M = ctx.num_indices
        N = ctx.num_cols
        # grad wrt S (since Y = tanh(S), dY/dS = 1 - tanh^2(S) = 1 - Y^2)
        grad_s = grad_output * (1 - Y * Y)

        # gradient wrt "other" is grad_s
        grad_other = grad_s

        # gradient wrt embedding weight
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros(
                ctx.weight_shape, device=grad_s.device, dtype=grad_s.dtype
            )

            # If scale_grad_by_freq, compute frequencies
            freq_ptr = None
            if ctx.scale_grad_by_freq:
                freq = torch.zeros(
                    (ctx.weight_shape[0],),
                    device=grad_s.device,
                    dtype=torch.int32,
                )
                BLOCK_IND = 256
                freq_grid = lambda meta: (triton.cdiv(M, BLOCK_IND),)
                _indice_freq_kernel[freq_grid](freq, indices, M, BLOCK_IND)
                freq_ptr = freq

            # Accumulate grad over embedding dimension
            BLOCK_SIZE = triton.next_power_of_2(N)
            grid = lambda meta: (M,)
            HAS_PADDING_IDX = ctx.padding_idx is not None

            _fused_embedding_backward_kernel[grid](
                grad_weight,
                grad_s,
                indices,
                ctx.padding_idx if HAS_PADDING_IDX else 0,
                HAS_PADDING_IDX,
                N,
                BLOCK_SIZE,
            )

            # Scale grad by freq if requested
            if ctx.scale_grad_by_freq and freq_ptr is not None:
                scale_grid = lambda meta: (triton.cdiv(ctx.weight_shape[0], 1),)
                _grad_scale_kernel[scale_grid](
                    grad_weight,
                    freq_ptr,
                    ctx.weight_shape[0],
                    N,
                    BLOCK_SIZE,
                )

            if ctx.sparse:
                # Convert to sparse representation if requested
                # (simple coalesced approach)
                idx_uniq = indices.unique()
                rows = idx_uniq.long()
                grads = grad_weight.index_select(0, rows)
                indices_for_sparse = rows.unsqueeze(1).expand(-1, grads.shape[1])
                indices_for_sparse = indices_for_sparse.reshape(1, -1)
                values = grads.reshape(-1, grads.shape[1])
                size = grad_weight.size()
                grad_weight = torch.sparse_coo_tensor(indices_for_sparse, values, size)

        # no grad wrt input_indices
        # grad wrt weight -> grad_weight
        # grad wrt other -> grad_other
        return (None, grad_weight, grad_other, None, None, None, None, None)


def fused_embedding_add_tanh(
    input_indices,
    weight,
    other,
    *,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    out=None,
):
    result = _FusedEmbeddingAddTanhFn.apply(
        input_indices,
        weight,
        other,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
