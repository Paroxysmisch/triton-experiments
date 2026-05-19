(tmp) / N + eps)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        b = tl.load(weights_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        c = a / rms * b
        tl.store(output_ptr + cols, c, mask=mask)
