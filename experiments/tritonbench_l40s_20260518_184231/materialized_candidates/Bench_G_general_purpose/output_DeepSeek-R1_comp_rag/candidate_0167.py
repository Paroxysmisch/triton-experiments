import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x,  # Pointer to input tensor
    g,  # Pointer to gate tensor
    o,  # Pointer to output tensor
    h0,  # Pointer to initial hidden state
    ht,  # Pointer to final hidden state
    T: tl.constexpr,  # Time steps
    D: tl.constexpr,  # Feature dimension
    BD: tl.constexpr,  # Block dimension
    USE_INITIAL_STATE: tl.constexpr,  # Flag for initial state
    STORE_FINAL_STATE: tl.constexpr,  # Flag to store final state
):
    # Extract program IDs
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    # Initialize pointers for the current block
    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d

    # Initialize hidden state
    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)

    # Temporal loop
    for _ in range(T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = tl.exp(b_g) * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)
        
        # Move pointers to next time step
        p_x += D
        p_g += D
        p_o += D

    # Store final state if needed
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)

@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g, o, dx, dg, do, h0,
    T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    # Initialize pointers to last element
    p_g = g + (i_bh * T + T - 1) * D + o_d
    p_o = o + (i_bh * T + T - 2) * D + o_d
    p_dx = dx + (i_bh * T + T - 1) * D + o_d
    p_dg = dg + (i_bh * T + T - 1) * D + o_d
    p_do = do + (i_bh * T + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T-1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        
        # Load previous hidden state
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        # Gradient accumulation
        b_dh += b_do
        tl.store(p_dx, b_dh.to(p_dx.dtype.element_ty), mask=mask)
        
        # Compute gradients
        b_dg_val = b_dh * b_o
        tl.store(p_dg, b_dg_val.to(p_dg.dtype.element_ty), mask=mask)
        
        # Update hidden gradient
        b_dh = b_dh * tl.exp(b_g)

        # Move pointers backward
        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        final_state = x.new_empty(B, H, D) if output_final_state else None
        o = torch.empty_like(x)
        
        # Configure kernel launch parameters
        BD = 128  # Tuned block size
        grid = (triton.cdiv(D, BD), B * H)
        
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            T, D, BD,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )
        ctx.save_for_backward(g, o, initial_state)
        ctx.BD = BD  # Save BD for backward pass
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape
        dx = torch.empty_like(o)
        dg = torch.empty_like(g)
        BD = ctx.BD  # Retrieve saved block dimension
        
        grid = (triton.cdiv(D, BD), B * H)
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state,
            T, D, BD,
            USE_INITIAL_STATE=initial_state is not None
        )
        return dx, dg, None, None

def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # Detach initial state for safety
    if initial_state is not None:
        initial_state = initial_state.detach()
    # Execute fused operation
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state
