cpp
__global__ void conv2d_forward_kernel(float* input, float* weight, float* output,
    int input_height, int input_width, int filter_height, int filter_width,
    int output_height, int output_width, int group_size) {

    // Calculate the global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the output
    if (x < output_width && y < output_height) {
        // Initialize the output value to 0
        output[y * output_width + x] = 0;

        // Perform the convolution
        for (int i = 0; i < filter_height; ++i) {
            for (int j = 0; j < filter_width; ++j) {
                for (int k = 0; k < group_size; ++k) {
                    int input_index = (y * filter_height + i) * input_width + x * group_size + k;
                    int weight_index = (k * group_size + i) * filter_width + j;
                    output[y * output_width + x] += input[input_index] * weight[weight_index];
                }
            }
        }
    }
}
