cpp
__device__ __inline__ uint32_t vmax(uint32_t a, uint32_t b) {
  return (a > b) ? a : b;
}

__device__ __inline__ uint32_t vmin(uint32_t a, uint32_t b) {
  return (a < b) ? a : b;
}

__global__ void _dequantize_rowwise(
  const float* x_ptr,
  const float* state_x,
  float* output_ptr,
  const float inv_127,
  const int n_elements,
  const int BLOCK_SIZE,
  const int P2
) {
  const int pid = blockIdx.x;
  const int block_start = pid * P2;
  const int start = block_start + threadIdx.x;
  const int end = vmin(block_start + P2, n_elements);

  float max_val = (start < n_elements) ? state_x[pid] : 0;

  for (int i = start; i < end; i += BLOCK_SIZE) {
    float val = x_ptr[i];
    output_ptr[i] = val * max_val * inv_127;
  }
}

void dequantize_rowwise(
  const float* x_ptr,
  const float* state_x,
  float* output_ptr,
  const int n_elements,
  const int BLOCK_SIZE
) {
  int P2 = 1;
  while (P2 < n_elements) P2 *= 2;

  dim3 grid_dim(n_elements);
  dim3 block_dim(BLOCK_SIZE);

  _dequantize_rowwise<<<grid_dim, block_dim>>>(
    x_ptr, state_x, output_ptr, 1.0f / 127.0f, n_elements, BLOCK_SIZE, P2
  );
}
