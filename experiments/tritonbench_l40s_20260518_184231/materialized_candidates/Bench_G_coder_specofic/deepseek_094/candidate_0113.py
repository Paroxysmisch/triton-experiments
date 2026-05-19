c++
__device__ void triton_red_fused_native_layer_norm_0(
    const float* primals_3, 
    float* out_ptr0, 
    int32_t* out_ptr1, 
    const int32_t* grid_stride, 
    const int32_t* block_stride, 
    int32_t* shared_mem, 
    int32_t thread_id) {
    
    int32_t block_id = blockIdx.x;
    int32_t thread_id = threadIdx.x;
    
    int32_t RBLOCK = block_stride[0];
    int32_t S = grid_stride[0];
    int32_t D = grid_stride[1];
    
    float* tmp3_mean = (float*)&shared_mem[0];
    float* tmp3_m2 = (float*)&shared_mem[S * D];
    int32_t* tmp3_weight = (int32_t*)&shared_mem[2 * S * D];
    
    float* tmp3 = &tmp3_mean[thread_id];
    float* tmp4 = &tmp3_m2[thread_id];
    int32_t* tmp5 = &tmp3_weight[thread_id];
    
    // Initialize shared memory
    if (thread_id < D) {
        tmp3_mean[thread_id] = 0.0f;
        tmp3_m2[thread_id] = 0.0f;
        tmp3_weight[thread_id] = 0;
    }
    
    // Synchronize to make sure all threads have initialized shared memory
    __syncthreads();
    
    // Perform reduction
    for (int32_t i = thread_id; i < S; i += RBLOCK) {
        float val = primals_3[block_id * S + i];
        tmp3[0] += val;
        tmp4[0] += val * val;
        tmp5[0] += 1;
    }
    
    // Synchronize to make sure all threads have completed the reduction
    __syncthreads();
    
    // Store results in shared memory
    if (thread_id < D) {
        out_ptr0[thread_id] = tmp3[0] / tmp5[0];
        out_ptr1[thread_id] = tmp4[0] / tmp5[0];
    }
}
