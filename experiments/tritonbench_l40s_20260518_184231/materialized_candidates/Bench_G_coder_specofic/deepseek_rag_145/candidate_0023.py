@triton.jit(do_not_specialize=('seq_len', ))
def rotary_embedding_kernel(
    Q,
    K,
    COS,
    SIN,
    seq_len,
    ...  #other parameters like strides
):
    """apply rotary on key AND query kernel."""

    #Load position offsets from given sequence length and constant strides
    pos_offset = seq_len // BLOCK * BLOCK + tl.program_id(0)

    #Load cosine and sine values according to positions
    loaded_cos = tl.load(COS + pos_offset)
    loaded_sin = tl.load(SIN + pos_offset)

    #Load Q and K tensor values
    loaded_q0 = tl.load(Q)
    loaded_q1 = tl.load(Q + stride_q)

    loaded_k0 = tl.load(K)
    loaded_k1 = tl.load(K + stride_k)

    #Apply cosine and sine to rotate Q and K
    rotated_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    rotated_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    rotated_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    rotated_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos

    #Store the results back to Q and K
    tl.store(Q, rotated_q0)
    tl.store(K, rotated_k0)

    #... similar operations for the rest of the elements
