def logit(input, eps=None, *, out=None):
    if out is None:
        out = input.new_empty(input.size())
    if eps is None:
        return out.logit(input)
    else:
        return out.logit(input, eps)
