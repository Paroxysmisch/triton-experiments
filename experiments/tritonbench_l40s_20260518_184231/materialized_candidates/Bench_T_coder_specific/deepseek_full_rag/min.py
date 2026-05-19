eps)
    tl.debug_barrier()

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        w = tl.load(weights_ptr + cols, mask=mask).to(DTYPE)
        normalized = a / rms
        result = normalized * w
        tl.store(output_ptr + cols, result, mask=mask)

@triton.jit
def min(input, dim=0, keepdim=False, *, out=None):
    # Implementation details omitted for brevity
    pass

def min(input, dim=0, keepdim=False, *, out=None):
    # Implementation details omitted for brevity
    pass
