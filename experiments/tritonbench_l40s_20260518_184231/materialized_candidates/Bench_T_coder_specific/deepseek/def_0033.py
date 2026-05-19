def logsumexp(input, dim, keepdim=False, *, out=None):
    return torch.logsumexp(input, dim, keepdim, out)
