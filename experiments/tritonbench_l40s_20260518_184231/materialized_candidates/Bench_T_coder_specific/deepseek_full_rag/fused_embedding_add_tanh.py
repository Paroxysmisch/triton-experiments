shape[:-1])
        N = weight.shape[1]
        BLOCK_SIZE = triton.next_power_of_2(N)
        NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)
        num_warps = 8
        if BLOCK_SIZE > 2047:
            num_warps = 16
        if BLOCK_SIZE > 4095:
            num_warps = 32

        out = torch.empty_like(indices, dtype=weight.dtype)
        with torch.cuda.device(weight.device):
            embedding_kernel[(M,)](
                out, indices, weight, N, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
            )

        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.indices = indices
        ctx.padding_idx = padding_idx
        ctx.sparse = sparse
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.NUM_BLOCKS = NUM_BLOCKS
        ctx.has_padding_idx = padding_idx is not None
        ctx.weight_stride = weight.stride(0)
        ctx.weight_size = N
        ctx.out_shape = indices.shape

        return out

    @staticmethod
    def backward(ctx, grad_out):
        indices = ctx.indices
        padding_idx = ctx.padding_idx
        scale_grad_by_freq = ctx.scale_grad_by_freq
        BLOCK_SIZE = ctx.BLOCK_SIZE
        NUM_BLOCKS = ctx.NUM_BLOCKS
        has_padding_idx = ctx.has_padding_idx
        weight_stride = ctx.weight_stride
        weight_size = ctx.weight_size
        out_shape = ctx.out_shape

        grad_in = torch.zeros(
            (indices.shape[0] * indices.shape[1], weight_size),
            dtype=grad_out.dtype,
            device=grad_out.device,
        )

        M = math.prod(indices.shape[:-1])
        N = weight.shape[1]
        BLOCK_SIZE = triton.next_power_of_2(N)
        NUM_BLOCKS = triton.cdiv(N, BLOCK_SIZE)
        num_warps = 8
        if BLOCK_SIZE > 2047:
            num_warps = 16
        if BLOCK_SIZE > 4095:
            num_warps = 32

        with torch.cuda.device(weight.device):
            embedding_backward_kernel[(M,)](
                grad_in,
                grad_out,
                indices,
                padding_idx,
                has_padding_idx,
                N,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        if scale_grad_by_freq:
            indices_freq = torch.zeros(
                grad_in.shape[0], dtype=grad_in.dtype, device=grad_in.device
            )
            INDICE_BLOCK_SIZE = triton.next_power_of_2(indices_freq.shape[0])
            NUM_INDICE_BLOCKS = triton.cdiv(indices_freq.shape[0], INDICE_BLOCK_SIZE)
            with torch.cuda.device(weight.device):
                indice_freq_kernel[(NUM_INDICE_BLOCKS,)](
                    indices_freq,
                    indices,
                    indices_freq.shape[0],
                    INDICE_BLOCK_SIZE,
                )
            with torch.cuda.device(weight.device):
                embedding_grad_scale_kernel[(indices_freq.shape[0],)](
                    grad_in, indices_freq, indices_freq.shape[0], weight_size
                )

        grad_weight = grad_in.view(-1, weight_stride)
        return grad_weight, None, None, None, None, None


def fused_embedding_add_tanh(
    input_indices, weight, other, *, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, out=None
):
    result = Embedding.apply(
        weight, input_indices, padding_idx, scale_grad_by_freq, sparse
    )
    result += other
    result = torch.tanh_(result)
    return result
