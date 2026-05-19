c++
__device__ __inline__
float my_shfl(float a, int b) {
  return __shfl(a, b);
}

__device__ __inline__
float my_shfl_down(float a, unsigned int b) {
  return __shfl_down(a, b);
}

__device__ __inline__
float my_shfl_up(float a, unsigned int b) {
  return __shfl_up(a, b);
}

__device__ __inline__
float my_shfl_xor(float a, int b) {
  return __shfl_xor(a, b);
}

__global__ void my_kernel(float* a, float* b) {
  __shared__ float shared_a[1024];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  shared_a[threadIdx.x] = a[idx];
  __syncthreads();

  b[idx] = shared_a[threadIdx.x] * 2.0f;
}
