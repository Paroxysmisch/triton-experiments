class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps=1e-5):
        # Prepare buffers for mean and variance
        mean, var = X.mean(dim=-1), X.var(dim=-1)
        # Normalize X
        Y = (X - mean[:, :, None]) / torch.sqrt(var[:, :, None] + eps)
        # Apply learned scale and shift
        Y = Y * W + B
        # Save necessary data for backward pass
        ctx.save_for_backward(Y, mean, var, W)
        return Y

    @staticmethod
    def backward(ctx, DY):
        # Load saved data
        Y, mean, var, W = ctx.saved_tensors
        # Compute gradient of inputs
        DX = DY * W
        DX = DX - (DX - Y).mean(dim=-1)[:, :, None]
        DX = DX / torch.sqrt(var[:, :, None] + eps)
        # Compute gradient of weights and biases
        DW = (DY * (Y - mean[:, :, None])).sum(dim=0)
        DB = DY.sum(dim=0)
        return DX, DW, DB
