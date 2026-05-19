c++
__device__ void chunk_global_cumsum_scalar_kernel(
    const int b,
    const int h,
    const int t,
    const int BT,
    const float* s,
    float* o,
    float* rs) {

  int bt = t / BT;
  int bh = b * H + h;

  float sum = 0.0f;
  for (int i = 0; i < BT; i++) {
    int idx = bh * T + bt * BT + i;
    float val = s[idx];
    sum += val;
    o[idx] = sum;
  }

  rs[bh] = sum;
}

void chunk_global_cumsum_scalar(
    const int B,
    const int H,
    const int T,
    const int BT,
    const float* s,
    float* o,
    float* rs,
    const cudaStream_t stream) {

  dim3 grid(B, H);
  dim3 block(T / BT);

  chunk_global_cumsum_scalar_kernel<<<grid, block, 0, stream>>>(B, H, T, BT, s, o, rs);
}
