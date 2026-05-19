N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0)
        tmp += a * a
    var = tl.sum(tmp, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0)
        normed = a * rstd
        tl.store(output_ptr + cols, normed, mask=mask)

    tl.store(weights_ptr + row, rstd)


def test_rms_norm():
    eps = 1e-5
    for _ in range(100):
        N = random.randint(1, 128)
        x = torch.randn(N, device='cuda')
        w = torch.randn(N, device='cuda')
        scale = torch.empty_like(x)
        y_ref = torch.ops.aten.rms_norm.default(x, w, eps)
        y = torch.empty_like(x)
        rms_norm[N, ](y, x, w, 1, N, eps, BLOCK_SIZE=32)
        assert torch.allclose(y, y_ref, atol=1e-6, rtol=0)


def test_rms_norm_large():
    eps = 1e-5
    N = 1024
    x = torch.randn(N, device='cuda')
    w = torch.randn(N, device='cuda')
    scale = torch.empty_like(x)
    y_ref = torch.ops.aten.rms_norm.default(x, w, eps)
    y = torch.empty_like(x)
    rms_norm[N, ](y, x, w, 1, N, eps, BLOCK_SIZE=32)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=0)


def test_rms_norm_broadcast():
    eps = 1e-5
    for _ in range(100):
        N = random.randint(1, 128)
        C = random.randint(1, 128)
        x = torch.randn((C, N), device='cuda')
        w = torch.randn(N, device='cuda')
        scale = torch.empty((C, N), device='cuda')
        y_ref = torch.ops.aten.rms_norm.default(x, w, eps)
        y = torch.empty_like(x)
        rms_norm[(C, 1), ](y, x, w, 1, N, eps, BLOCK_SIZE=32)
        assert torch.allclose(y, y_ref, atol=1e-6, rtol=0)


def test_rms_norm_grad():
    for _ in range(100):
        N = random.randint(1, 128)
        x = torch.randn(N, device='cuda', dtype=torch.float64, requires_grad=True)
        w = torch.randn(N, device='cuda', dtype=torch.float64, requires_grad=True)
        y = torch.ops.aten.rms_norm.default(x, w, 1e-5)
        torch.sum(y).backward()
        x_grad_ref = x.grad.clone()
        w_grad_ref = w.grad.clone()
        x.grad = None
        w.grad = None
        rms_norm_grad[(N, 1), ](x, w, x_grad, w_grad, 1, N, 1e-5, BLOCK_SIZE=32)
        assert torch.allclose(x.grad, x_grad_ref, atol=1e-6, rtol=0)
        assert torch.allclose(w.grad, w_grad_ref, atol=1e-6, rtol=0)
