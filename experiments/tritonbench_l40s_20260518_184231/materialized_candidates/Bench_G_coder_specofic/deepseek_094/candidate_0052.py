cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace cooperative_groups;

__global__ void swizzle_tile(...) {
    // Implementation
}

__global__ void linear_tile(...) {
    // Implementation
}

__global__ void mac_loop(...) {
    // Implementation
}

__global__ void first_wave(...) {
    // Implementation
}

__global__ void full_tiles(...) {
    // Implementation
}

class matmul {
public:
    matmul(...) {
        // Initialize variables
    }

    void _call(...) {
        // Set up grid, allocate memory, call kernels
    }

    void forward(...) {
        // Wrapper for _call
    }

private:
    // Variables
};
