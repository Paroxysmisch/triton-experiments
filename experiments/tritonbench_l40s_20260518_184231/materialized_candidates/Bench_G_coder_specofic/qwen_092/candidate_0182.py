triton
// fused_recurrent_hgrn_fwd_kernel
@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x: pointer[f32, "rw"],
    g: pointer[f32, "rw"],
    o: pointer[f32, "rw"],
    h0: pointer[f32, "rw"],
    ht: pointer[f32, "rw"],
    T: i32,
    D: i32,
    BD: i32,
    USE_INITIAL_STATE: i32,
    STORE_FINAL_STATE: i32,
    BLOCK_SIZE: i32 = 256
):
    pid = triton.program_id(0)
    grid_size = triton.cdiv(T, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + triton.block_id(0)
    if idx < T:
        b_x = x[idx * D + BD]
        b_g = g[idx * D + BD]
        b_h = h0[BD] if USE_INITIAL_STATE == 1 else 0.0
        b_h = b_g * b_h + b_x
        o[idx * D + BD] = b_h
        if STORE_FINAL_STATE == 1:
            ht[BD] = b_h

// fused_recurrent_hgrn_bwd_kernel
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    x: pointer[f32, "rw"],
    g: pointer[f32, "rw"],
    o: pointer[f32, "rw"],
    do: pointer[f32, "rw"],
    dx: pointer[f32, "rw"],
    dg: pointer[f32, "rw"],
    dh: pointer[f32, "rw"],
    T: i32,
    D: i32,
    BD: i32,
    BLOCK_SIZE: i32 = 256
):
    pid = triton.program_id(0)
    grid_size = triton.cdiv(T, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + triton.block_id(0)
    if idx < T:
        b_do = do[idx * D + BD]
        b_dh = dh[idx * D + BD]
        b_dx = b_dh
        b_dg = b_dh * o[idx * D + BD]
        b_dh = b_dh * g[idx * D + BD]
        dx[idx * D + BD] = b_dx
        dg[idx * D + BD] = b_dg

// FusedRecurrentHGRNFunction
class FusedRecurrentHGRNFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, g, o, h0=None, ht=None, T=None, D=None, BD=None, USE_INITIAL_STATE=0, STORE_FINAL_STATE=0):
        if h0 is None:
            h0 = torch.zeros_like(o[:, :BD])
        if ht is None:
            ht = torch.zeros_like(o[:, :BD])
        if T is None:
            T = o.shape[0]
        if D is None:
            D = o.shape[1]
        if BD is None:
            BD = o.shape[2]
        fused_recurrent_hgrn_fwd_kernel[grid_size, BLOCK_SIZE](x, g, o, h0, ht, T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE)
        ctx.save_for_backward(x, g, o, h0, ht)
        return o, ht

    @staticmethod
    def backward(ctx, do, dh):
        x, g, o, h0, ht = ctx.saved_tensors
        dx = torch.zeros_like(x)
        dg = torch.zeros_like(g)
        fused_recurrent_hgrn_bwd_kernel[grid_size, BLOCK_SIZE](x, g, o, do, dx, dg, dh, T, D, BD)
        return dx, dg, None, None, None, None, None, None, None, None

// fused_recurrent_hgrn
def fused_recurrent_hgrn(x, g, o, h0=None, ht=None, T=None, D=None, BD=None, USE_INITIAL_STATE=0, STORE_FINAL_STATE=0):
    o, ht = FusedRecurrentHGRNFunction.apply(x, g, o, h0, ht, T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE)
    return o, ht
