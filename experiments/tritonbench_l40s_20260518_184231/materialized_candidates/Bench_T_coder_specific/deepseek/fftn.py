import numpy as np

def fftn(input, s=None, dim=None, norm=None):
    # Default values for s and dim
    if s is None:
        s = [input.size(d) for d in dim]
    if dim is None:
        dim = tuple(range(input.dim()))

    # Perform the Fourier transform
    output = np.fft.fftn(input, s, dim, norm)

    return output
