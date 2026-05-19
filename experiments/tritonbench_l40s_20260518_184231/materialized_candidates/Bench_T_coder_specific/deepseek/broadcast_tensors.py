def broadcast_tensors(*tensors):
    # Clone the tensors to ensure they are not modified in-place
    tensors = [t.clone() for t in tensors]

    # Use Triton's broadcast_tensors function
    result = tl.broadcast_tensors(*tensors)

    return result
