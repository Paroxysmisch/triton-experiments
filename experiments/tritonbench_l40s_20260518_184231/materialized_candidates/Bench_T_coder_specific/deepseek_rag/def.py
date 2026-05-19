.
    - **pivot** (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`.
    - **out** (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
Math: LU Decomposition:
A = P L U
- P is a permutation matrix.
- L is a lower triangular matrix with unit diagonal elements.
- U is an upper triangular matrix.
Solving Linear Systems:
For each b_i in B, compute x_i by:
- Apply permutation: b'_i = P^T b_i
- Solve L y_i = b'_i
- Solve U x_i = y_i
other: This function efficiently reuses the LU decomposition of A to solve multiple linear systems with different right-hand sides. If `pivot=False`, no permutation is applied. Supports batch dimensions.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate.
<|system|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves multiple linear systems with the same coefficient matrix using LU decomposition. Given a square matrix A and multiple right-hand side vectors B, this function computes the solutions X to the linear systems A X = B by performing the LU decomposition of A and reusing it to solve for multiple right-hand sides efficiently. Supports batch dimensions.
Wrapper Entry Information: def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor

    - **A** (Tensor): Coefficient matrix of shape `(*, n, n)`, where `*` is zero or more batch dimensions.
    - **Bs** (Tensor): Right-hand side tensor of shape `(*, n, k)`, where `k` is the number of right-hand sides.
    - **pivot** (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`.
    - **out** (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
Math: LU Decomposition:
A = P L U
- P is a permutation matrix.
- L is a lower triangular matrix with unit diagonal elements.
- U is an upper triangular matrix.
Solving Linear Systems:
For each b_i in B, compute x_i by:
- Apply permutation: b'_i = P^T b_i
- Solve L y_i = b'_i
- Solve U x_i = y_i
other: This function efficiently reuses the LU decomposition of A to solve multiple linear Systems with different right-hand sides. If `pivot=False`, no permutation is applied. Supports batch dimensions.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate.
<|end-system|>
"""


def test_generate_triton_code_with_wrapper_function():
    func_inputs = [
        {
            "func_name": "solve_multiple_lu",
            "func_description": "Solves multiple linear systems with the same coefficient matrix using LU decomposition. Given a square matrix A and multiple right-hand side vectors B, this function computes the solutions X to the linear systems A X = B by performing the LU decomposition of A and reusing it to solve for multiple right-hand sides efficiently. Supports batch dimensions.",
            "func_parameters": [
                {"name": "A", "type": "Tensor", "description": "Coefficient matrix of shape `(*, n, n)`, where `*` is zero or more batch dimensions."},
                {"name": "Bs", "type": "Tensor", "description": "Right-hand side tensor of shape `(*, n, k)`, where `k` is the number of right-hand sides."},
                {"name": "pivot", "type": "bool", "description": "Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`).", "default": "True"},
                {"name": "out", "type": "Tensor", "description": "Output tensor. Ignored if `None`.", "default": "None"},
            ],
            "math": "LU Decomposition: A = P L U. Solving Linear Systems: For each b_i in B, compute x_i by: - Apply permutation: b'_i = P^T b_i - Solve L y_i = b'_i - Solve U x_i = y_i other: This function efficiently reuses the LU decomposition of A to solve multiple linear systems with different right-hand sides. If `pivot=False`, no permutation is applied. Supports batch dimensions.",
            "other": "After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate.",
        }
    ]

    generated_code = generate_triton_code_with_wrapper_function(func_inputs)
    expected_code = """
<|user|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves multiple linear systems with the same coefficient matrix using LU decomposition. Given a square matrix A and multiple right-hand side vectors B, this function computes the solutions X to the linear systems A X = B by performing the LU decomposition of A and reusing it to solve for multiple right-hand sides efficiently. Supports batch dimensions.
Wrapper Entry Information: def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor

    - **A** (Tensor): Coefficient matrix of shape `(*, n, n)`, where `*` is zero or more batch dimensions.
    - **Bs** (Tensor): Right-hand side tensor of shape `(*, n, k)`, where `k` is the number of right-hand sides.
    - **pivot** (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`.
    - **out** (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
Math: LU Decomposition:
A = P L U
- P is a permutation matrix.
- L is a lower triangular matrix with unit diagonal elements.
- U is an upper triangular matrix.
Solving Linear Systems:
For each b_i in B, compute x_i by:
- Apply permutation: b'_i = P^T b_i
- Solve L y_i = b'_i
- Solve U x_i = y_i
other: This function efficiently reuses the LU decomposition of A to solve multiple linear systems with different right-hand sides. If `pivot=False`, no permutation is applied. Supports batch dimensions.
After generation, verify if the Triton wrapper aligns with the provided func_inputs. If not, regenerate.
<|system|>
You are an expert in Trion programming, capable of writing corresponding Triton kernels and wrapper functions based on functional descriptions and function parameters. Ensure that the wrapper function fully corresponds to the provided function information.
Functional Description: Solves multiple linear systems with the same coefficient matrix using LU decomposition. Given a square matrix A and multiple right-hand side vectors B, this function computes the solutions X to the linear systems A X = B by performing the LU decomposition of A and reusing it to solve for multiple right-hand sides efficiently. Supports batch dimensions.
Wrapper Entry Information: def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor

    - **A** (Tensor): Coefficient matrix of shape `(*, n, n)`, where `*` is zero or more batch dimensions.
    - **Bs** (Tensor): Right-hand side tensor of shape `(*, n, k)`, where `k` is the number of right-hand sides.
    - **pivot** (bool, optional): Controls whether to compute the LU decomposition with partial pivoting (`True`) or without pivoting (`False`). Default: `True`.
    - **out** (T
