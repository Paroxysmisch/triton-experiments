tl.debug_barrier()
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        b = tl.load(weights_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        c = a / rms * b
        tl.store(output_ptr + cols, c, mask=mask)

def rms_norm_wrapper(x, weight, eps):
    """
    Wrapper function for the Triton kernel `rms_norm`
    """
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    N = x.shape[-1]
    grid = (x.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm[grid,](x, output, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, DTYPE=tl.float16)
    return output

def rms_norm_fused_wrapper(x, weight, eps):
    """
    Wrapper function for the Triton kernel `rms_norm`
    """
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    N = x.shape[-1]
    grid = (x.shape[0],)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm[grid,](x, output, weight, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, DTYPE=tl.float16)
    return output
