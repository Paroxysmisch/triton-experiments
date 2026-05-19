import triton
import triton.language as tl

@triton.jit
def _softmax(x_ptr, y_ptr, n_rows, n_cols, depth, is_log):
    # your code here
    pass

def softmax(x, is_log=False):
    # input validation
    # tensor preparation
    # call _softmax with appropriate parameters
    pass

@triton.jit
def _softmax_backward(dy_ptr, dx_ptr, y_ptr, n_rows, n_cols, depth, is_log):
    # your code here
    pass

def softmax_backward(dy, y, is_log=False):
    # input validation
    # tensor preparation
    # call _softmax_backward with appropriate parameters
    pass
