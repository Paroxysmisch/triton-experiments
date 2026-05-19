import triton as tl

# Define the kernel
@tl.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    DestLoc,
    O_nope,
    O_rope,
    BLOCK_DMODEL_NOPE,
    BLOCK_DMODEL_ROPE,
):
    # Prepare aligned dimensions
    aligned_nope = (BLOCK_DMODEL_NOPE + 127) // 128 * 128
    aligned_rope = (BLOCK_DMODEL_ROPE + 127) // 128 * 128

    # Calculate strides
    stride_nope = KV_nope.stride(-2)
    stride_rope = KV_rope.stride(-2)
    stride_o_nope = O_nope.stride(-2)
    stride_o_rope = O_rope.stride(-2)

    # Iterate over each index in the sequence
    for idx in tl.arange(0, DestLoc.numel()):
        # Get the index
        i = DestLoc[idx]

        # Load elements from the source tensors
        v_nope = tl.load(KV_nope + i * stride_nope)
        v_rope = tl.load(KV_rope + i * stride_rope)

        # Store them in the corresponding position in the output tensors
        tl.store(O_nope + i * stride_o_nope, v_nope)
        tl.store(O_rope + i * stride_o_rope, v_rope)

# Define the wrapper
def destindex_copy_kv(
    KV_nope,
    KV_rope,
    DestLoc,
    O_nope,
    O_rope,
    num_warps=64,
    num_stages=4,
    BLOCK_DMODEL_NOPE=128,
    BLOCK_DMODEL_ROPE=128,
):
    # Check shapes
    assert KV_nope.shape == O_nope.shape
    assert KV_rope.shape == O_rope.shape

    # Prepare grid size
    grid = (
        tl.cdiv(KV_nope.numel(), num_warps * BLOCK_DMODEL_NOPE),
        num_stages,
    )

    # Invoke the kernel
    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
        BLOCK_DMODEL_NOPE,
        BLOCK_DMODEL_ROPE,
    )
