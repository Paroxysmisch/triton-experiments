cpp
__device__
void permute_copy_kernel(
  const float* input,
  float* output,
  const int* dims,
  const int* input_strides,
  const int* output_strides,
  const int num_dims
) {
  int output_idx = 0;
  for (int i = 0; i < num_dims; ++i) {
    output_idx += dims[i] * output_strides[i];
  }
  for (int i = 0; i < num_dims; ++i) {
    output_idx = (output_idx - dims[i] * output_strides[i]) / input_strides[i];
  }
  output[output_idx] = input[output_idx];
}
