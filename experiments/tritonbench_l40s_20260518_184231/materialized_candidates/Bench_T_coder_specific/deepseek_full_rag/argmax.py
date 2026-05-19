_policy='evict_last')
        w = tl.load(weights_ptr + cols, mask=mask, other=0.0)
        y = x / rms * w
        tl.store(output_ptr + cols, y, mask=mask)


@triton.jit
def argmax(input, dim, keepdim):
    if dim is None:
        return tl.argmax(input)
    else:
        return tl.argmax(input, dim=dim, keepdim=keepdim)

def argmax(input, dim=None, keepdim=False):
    if dim is None:
        return torch.argmax(input)
    else:
        if keepdim:
            return torch.argmax(input, dim=dim, keepdim=True)
        else:
            return torch.argmax(input, dim=dim, keepdim=False)
