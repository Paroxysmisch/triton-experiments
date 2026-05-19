class FlashAttention:
    def __init__(self):
        pass

    def _fwd_kernel(self, ...):
        ...
        return o, lse, softmax_scale

    def flash_attn_triton(self, q, k, v, bias=None, causal=False, softmax_scale=None):
        ...
        return o, lse, softmax_scale
