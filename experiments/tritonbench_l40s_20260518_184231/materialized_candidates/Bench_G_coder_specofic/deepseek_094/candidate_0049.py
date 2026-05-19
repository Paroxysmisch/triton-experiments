c
__device__
int add(int a, int b) {
    return a + b;
}

__global__
void add_kernel(int* c, const int* a, const int* b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = add(a[i], b[i]);
}
