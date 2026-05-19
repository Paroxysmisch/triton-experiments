import triton as tl
import numpy as np

@tl.autotune(key="block_size")
def chunk_global_cumsum_vector_kernel(
    s, z, BT, BS, stride=(1, 1, 1, 1),
):
    pid = tl.program_id(axis=0)
    bid = tl.block_id(axis=0)
    tid = tl.thread_id(axis=0)

    m_s = tl.lower_triangular_mask(BT, tid)
    b_s = tl.make_block_ptr(s, (pid, bid, 0, 0), stride)
    b_s = tl.load(b_s, mask=m_s, other=0.0, ptr_type=tl.uint32)
    b_c = tl.dot(b_s, m_s)
    b_z = tl.load(z, (pid, bid, 0, 0), stride)
    b_z = tl.max(b_z, b_c)
    tl.store(z, b_z, (pid, bid, 0, 0), stride)

def chunk_global_cumsum_vector(s, BT, BS):
    S = s.shape
    z = tl.zeros(S, dtype=s.dtype)
    grid = lambda meta: (meta["B"], meta["T"], meta["S"])
    chunk_global_cumsum_vector_kernel[grid](s, z, BT, BS)
    return z
