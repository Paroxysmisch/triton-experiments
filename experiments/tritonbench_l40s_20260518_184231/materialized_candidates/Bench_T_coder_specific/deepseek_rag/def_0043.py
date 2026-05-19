input = torch.randn(10)
beta = 2.

output_forward = _smelu_triton_forward(input, beta)
output_backward = _smelu_triton_backward(input, beta)
