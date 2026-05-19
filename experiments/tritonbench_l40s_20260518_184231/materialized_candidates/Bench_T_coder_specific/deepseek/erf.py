import triton
import numpy as np

# create a tensor with some data
input_data = np.random.rand(1000).astype(np.float32)
input_tensor = triton.language.current.Tensor(input_data)

# call the erf function
output_tensor = triton.language.current.erf(input_tensor)

# retrieve the data from the output tensor
output_data = output_tensor.numpy()
