import triton as tl

def polygamma(n, input, *, out=None):
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    return tl.polygamma(n, input, out=out)
