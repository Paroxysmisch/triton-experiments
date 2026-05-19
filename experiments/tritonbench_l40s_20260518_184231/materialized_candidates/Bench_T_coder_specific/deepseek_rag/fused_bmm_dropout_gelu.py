def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, *, out=None):
    # Check input shapes
    assert input1.shape[0] == input2.shape[0]
    assert input1.shape[1] == input2.shape[2]

    # Perform batch matrix multiplication
    output = torch.bmm(input1, input2)

    # Apply dropout
    if training:
        output = torch.nn.functional.dropout(output, p=p, training=True)
    else:
        output = output * (1 - p)

    # Apply GELU activation
    output = torch.nn.functional.gelu(output)

    return output
