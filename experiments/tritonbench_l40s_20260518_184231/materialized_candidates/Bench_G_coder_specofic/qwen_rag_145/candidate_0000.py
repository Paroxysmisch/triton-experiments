class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        attn = torch.bmm(q, k.transpose(1, 2)) / scale
        ctx.save_for_backward(q, k, v, attn)
        return attn

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attn = ctx.saved_tensors
        dk = torch.bmm(q.transpose(1, 2), grad_output)
        dv = torch.bmm(grad_output, k)
        dq = torch.bmm(grad_output, v.transpose(1, 2))
        return dq, dk, dv, None

attention = ScaledDotProductAttention.apply
