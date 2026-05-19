occupancy * num_warps
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    # Run kernel
    kernel.run(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
               num_warps=num_warps, grid=(num_programs, ))
    return y

softmax_output = softmax(input_tensor)
"""

# Test case 1:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 2}
expected_output = [[4, 2], [0, 2]]

# Test case 2:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 0.5}
expected_output = [[1.224744871391589, 0.44721359549995796], [0, 0.8944271909999159]]

# Test case 3:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": 2}
expected_output = [[8.0, 2.0+2.0j], [2.0+2.0j, 1.0]]

# Test case 4:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": 0.5}
expected_output = [[1.4142135623730951+0.7071067811865476j, 0.8414709848078965+0.5403023058681398j], [0.8414709848078965+0.5403023058681398j, 0.7071067811865476]]

# Test case 5:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": 2}
expected_output = [[37.53, 56.25, 75.96], [105.97, 149.13, 194.68], [254.24, 342.39, 443.16]]

# Test case 6:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": 0.5}
expected_output = [[2.17, 2.83, 3.49], [3.83, 4.68, 5.6], [6.86, 8.64, 10.6]]

# Test case 7:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 1.5}
expected_output = [[2.71828182846, 1.831563889], [0, 2.71828182846]]

# Test case 8:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": 1.5}
expected_output = [[4.48874457028, 2.8+1.8j], [2.8+1.8j, 2.236067977]]

# Test case 9:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": 1.5}
expected_output = [[6.28, 10.48, 15.31], [24.18, 35.51, 50.68], [90.49, 129.86, 185.93]]

# Test case 10:
func_inputs = {"A": [[2, 1], [0, 1]], "k": -1}
expected_output = [[0.5, 2], [1, 1]]

# Test case 11:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": -1}
expected_output = [[0.40824829046386307, 0.40824829046386307], [0.7071067811865476, 0.40824829046386307]]

# Test case 12:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": -1}
expected_output = [[0.19035024027722996, 0.15275490428004684, 0.1319190806900202], [0.06637578184650516, 0.06064999604824179, 0.05617378259880554], [0.04332090370807474, 0.03920135769650498, 0.03607946833921498]]

# Test case 13:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 0}
expected_output = [[1, 1], [1, 1]]

# Test case 14:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": 0}
expected_output = [[1, 1], [1, 1]]

# Test case 15:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": 0}
expected_output = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

# Test case 16:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 1}
expected_output = [[2, 1], [0, 1]]

# Test case 17:
func_inputs = {"A": [[2, 1+1j], [1+1j, 1]], "k": 1}
expected_output = [[2+1j, 1+1j], [1, 1]]

# Test case 18:
func_inputs = {"A": [[1, 2, 3], [4, 5, 6], [7, 8, 10]], "k": 1}
expected_output = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]

# Test case 19:
func_inputs = {"A": [[2, 1], [0, 1]], "k": 2.5}
expected_output = [[4.4721359549995796, 2.2360679774997898], [0, 2.2360679774997898]]

# Test case
