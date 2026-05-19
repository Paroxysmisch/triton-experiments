cpp
__device__ float add(float a, float b) {
  return a + b;
}

__global__ void add_kernel(float* a, float* b, float* c, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    c[idx] = add(a[idx], b[idx]);
  }
}
