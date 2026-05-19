@triton.jit
def signbit_bitwise_and(input_ptr, other_ptr, output_ptr, n):
    input = triton.mem[input_ptr]
    other = triton.mem[other_ptr]
    output = triton.mem[output_ptr]

    for i in range(n):
        signbit_result = input[i] < 0
        bitwise_and_result = input[i] & other[i]
        output[i] = signbit_result, bitwise_and_result
