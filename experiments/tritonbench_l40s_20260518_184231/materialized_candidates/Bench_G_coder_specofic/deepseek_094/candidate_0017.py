c++
#include <cuda_runtime.h>
#include <triton/triton.h>

#define CHECK_CUDA(call) {                                                  \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),        \
               __FILE__, __LINE__);                                         \
        exit(0);                                                            \
    }                                                                       \
}

#define CHECK_TRITON(call) {                                                 \
    triton::client::ClientException e = call;                                \
    if (e.what() != nullptr) {                                               \
        printf("Triton Error: %s at %s:%d\n", e.what(), __FILE__, __LINE__);  \
        exit(0);                                                            \
    }                                                                       \
}

// define your kernel here
__global__ void _attn_fwd_inner(...) {
    // your code here
}

// define your wrapper function here
void _attn_fwd(...) {
    // your code here
}

// define your wrapper function here
void forward(...) {
    // your code here
}

int main() {
    // Initialize Triton client
    triton::client::Client client("localhost:8000");

    // Define input tensors
    // ...

    // Define output tensor
    // ...

    // Call forward function
    forward(input_tensors, output_tensor);

    // Get output data
    // ...

    return 0;
}
