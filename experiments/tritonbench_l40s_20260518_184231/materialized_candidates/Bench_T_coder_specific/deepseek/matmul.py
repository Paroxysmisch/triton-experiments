def matmul(input, other, out=None):
    if out is not None:
        return torch.matmul(input, other, out=out)
    else:
        return torch.matmul(input, other)
