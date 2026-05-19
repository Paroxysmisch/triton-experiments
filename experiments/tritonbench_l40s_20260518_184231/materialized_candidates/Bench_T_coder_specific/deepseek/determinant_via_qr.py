import triton as tl
import numpy as np

def determinant_via_qr(A, *, mode='reduced', out=None):
    # Convert inputs to Triton tensors
    A = tl.convert_to_tensor(A)
    mode = tl.convert_to_tensor(mode)
    out = tl.convert_to_tensor(out) if out is not None else None

    # Perform QR decomposition
    Q, R = tl.linalg.qr(A, mode=mode)

    # Compute determinant
    det_Q, _ = tl.linalg.det(Q)
    det_R = tl.prod(tl.diagonal(R))
    det_A = det_Q * det_R

    # If an output tensor is provided, store the result there
    if out is not None:
        out.set(det_A)

    return det_A
