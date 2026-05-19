import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr, 
    mask_ptr, scale, num_heads, 
    seq_length, head_size, 
    output_size_per_head, 
    IFCOND, STORE
):
    # Setup memory pointers
    p_q = q_ptr + tl.arange(0, seq_length) * head_size
    p_k = k_ptr + tl.arange(0, seq_length) * head_size
    p_v = v_ptr + tl.arange(0, seq_length) * head_size
    p_h = h_ptr + tl.arange(0, seq_length) * output_size_per_head
    p_o = o_ptr + tl.arange(0, seq_length) * output_size_per_head

    # Initialize b_h as a zero matrix
    b_h = tl.zeros((seq_length, output_size_per_head), dtype=tl.float32)

    # Iterate over divided blocks of the sequence length
    for i in range(0, seq_length, BT):
        # Load blocks of the query and key tensors
        q = tl.load(p_q + i, mask=mask_ptr, other=-1e4)
        k = tl.load(p_k + i, mask=mask_ptr, other=-1e4)

        # Compute scaled dot-product attention
        b_s = tl.softmax((q @ k.T) * scale, axis=-1)

        # Load block of the value tensor
        v = tl.load(p_v + i, mask=mask_ptr, other=-1e4)

        # Weight the block of the value tensor by the attention scores
        b_o = b_s @ v

        # Apply update to the intermediate b_h tensor
        if IFCOND:
            b_h = b_h + b_o
        else:
            b_h = b_o

        # Optionally store the intermediate b_h tensor
        if STORE:
            tl.store(p_h + i, b_h)

    # Store the computed output tensor
    tl.store(p_o, b_h)


class AttentionFunction:
    def __init__(self, num_heads, head_size, output_size_per_head):
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_size_per_head = output_size_per_head

    def forward(self, q, k, v, h, o, mask, scale, IFCOND, STORE):
        # Initialize output tensor
        o = tl.zeros_like(h)

        # Compute scaling based on head dimension
        scale = 1.0 / math.sqrt(self.head_size)

        # Setup grid size, number of warps, and stages
        grid = lambda meta: (meta['serial_granularity'],)
        num_warps = 1
        num_stages = 1

        # Call the Triton kernel
        attention_fwd_kernel[grid](
            q, k, v, h, o, 
            mask, scale, self.num_heads, 
            q.shape[1], self.head_size, 
            self.output_size_per_head, 
            IFCOND, STORE
        )

        return o
