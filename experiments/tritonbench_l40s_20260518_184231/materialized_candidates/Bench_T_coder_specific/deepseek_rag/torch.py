def permute_copy(input, dims):
    return input.permute(*dims).clone()
