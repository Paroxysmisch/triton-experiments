def softmax(input):
    logsumexp_output = logsumexp(input)
    return input.exp() / logsumexp_output
