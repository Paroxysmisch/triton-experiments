@triton.jit
def _fwd_recurrence(
    S, d, O,
    NUM_HEAD: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr,
    last_kv=None
):
    i, j, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)

    s_offset_k, s_offset_v = j * D_MODEL_K + i * D_MODEL_K * NUM_HEAD, j * BLOCK_MODEL_K + i * BLOCK_MODEL_K * NUM_HEAD

    K = tl.load(S + s_offset_k, boundary_check=(0, 0))
    V = tl.load(S + s_offset_v, boundary_check=(0, 0))
    decay = tl.load(d + i * NUM_HEAD + j)

    if last_kv is not None:
        last_K = last_kv[0][s_offset_k]
        last_V = last_kv[0][s_offset_v]
        K += decay[:, None] * (last_K - K)
        V += decay[:, None] * (last_V - V)

    O_update = tl.dot(K, V)
    tl.store(O + s_offset_k, O_update)

@triton.jit
def _bwd_recurrence(
    S, d, DI, DG, DL, DS,
    NUM_HEAD: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_MODEL_K: tl.constexpr,
    D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr
):
    i, j, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)

    s_offset_k, s_offset_v = j * D_MODEL_K + i * D_MODEL_K * NUM_HEAD, j * BLOCK_MODEL_K + i * BLOCK_MODEL_K * NUM_HEAD

    k_grad = tl.load(DI + s_offset_k, boundary_check=(0, 0))
    v_grad = tl.load(DG + s_offset_v, boundary_check=(0, 0))
    decay = tl.load(d + i * NUM_HEAD + j)

    tl.store(DG + s_offset_k, k_grad * decay + v_grad)
    tl.store(DL + s_offset_v, v_grad * decay)
    tl.store(DS + s_offset_k, decay)

class ChunkGateRecurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cross_decay, last_kv=None):
        O = torch.empty_like(kv)
        _fwd_recurrence[...](kv, cross_decay, O, last_kv=last_kv)
        return O

    @staticmethod
    def backward(ctx, DO):
        DI = torch.empty_like(DO)
        DG = torch.empty_like(DO)
        DL = torch.empty_like(DO)
        DS = torch.empty_like(DO)
        _bwd_recurrence[...](DO, DI, DG, DL, DS)
        return DI, DG, DL, DS
