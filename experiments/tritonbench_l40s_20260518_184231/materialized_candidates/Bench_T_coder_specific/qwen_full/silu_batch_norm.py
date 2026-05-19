import torch
import triton
import triton.language as tl

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def silu_bmm(out, a, b, running_mean, running_var, weight, bias, N, HW, C, momentum, eps, train):
    pid = tl.program_id(0)
    if pid < N:
        cols = tl.arange(0, HW)
        batch_mean = tl.sum(a[pid, :, :], axis=1) / HW
        var = tl.sum((a[pid, :, :] - batch_mean[:, None]) * (a[pid, :, :] - batch_mean[:, None]), axis=1) / HW
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * var
        x_hat = (a[pid, :, :] - batch_mean[:, None]) * (1 / tl.sqrt(var[:, None] + eps))
        out1 = tl.sum((b[pid, :, :] - running_mean[:, None]) * (x_hat), axis=1)
        if train:
            tl.store(running_mean + pid * C + cols, running_mean)
            tl.store(running_var + pid * C + cols, running_var)
        w = tl.load(weight + cols)
        out2 = tl.load(bias + cols)
        out[pid, :] = silu(out1 * w + out2)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    assert input.is_contiguous()
    assert running_mean.is_contiguous()
    assert running_var.is_contiguous()
    if weight is not None:
        assert weight.is_contiguous()
    if bias is not None:
        assert bias.is_contiguous()
    input = input.view(input.size(0), -1)
    N, HW = input.size()
    C = HW // N
    out = torch.empty_like(input)
    running_mean = running_mean.view(-1)
    running_var = running_var.view(-1)
    if weight is not None:
        weight = weight.view(-1)
    if bias is not None:
        bias = bias.view(-1)
    grid = lambda META: (N, )
    silu_bmm[grid](out, input, out, running_mean, running_var, weight, bias, N, HW, C, momentum, eps, not training)
    return out.view_as(input)
