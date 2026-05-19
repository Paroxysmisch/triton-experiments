input_tensor = torch.randn(3, 3)
print("Input Tensor:")
print(input_tensor)

output_tensor = square_matrix(input_tensor)
print("Square of Tensor:")
print(output_tensor)

output_tensor = mean_of_squares_matrix(input_tensor, 1e-8)
print("Mean of Squares of Tensor:")
print(output_tensor)

output_tensor = rms_matrix(input_tensor, 1e-8)
print("RMS of Tensor:")
print(output_tensor)

weights_tensor = torch.randn(3, 3)
output_tensor = rms_norm_matrix(input_tensor, weights_tensor, 1e-8)
print("RMS Norm of Tensor:")
print(output_tensor)

output_tensor = argmax(input_tensor, dim=1, keepdim=True)
print("Argmax of Tensor:")
print(output_tensor)
