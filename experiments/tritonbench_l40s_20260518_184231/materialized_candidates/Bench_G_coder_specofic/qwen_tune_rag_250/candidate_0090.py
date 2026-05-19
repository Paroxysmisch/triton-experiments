import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

logits_strides = (1, 0)
v_strides = (0, 1)
out_strides = (1, 0)
B_Loc_stride = 0
B_Start_Loc_stride = 0
B_Seqlen_stride = 0

class _token_softmax_reducev_fwd(Function):
    @staticmethod
    def forward(ctx, logits: Tensor, v: Tensor, B_Loc: Tensor, B_Start_Loc: Tensor, B_Seqlen: Tensor, block_dmodel: int, ignore_index: int):
        batch, n_heads, seq_len, d_model = logits.shape
        _, _, v_seq_len, v_head_dim = v.shape
        assert d_model == block_dmodel
        assert v_head_dim == d_model
        assert B_Loc.shape == B_Start_Loc.shape
        assert B_Start_Loc.shape == B_Seqlen.shape
        assert v_seq_len == B_Seqlen.max().item()

        ignore_index = -100 if ignore_index == -1 else ignore_index
        logits_strides = logits.stride()
        v_strides = v.stride()
        out_strides = logits.stride()
        B_Loc_stride = B_Loc.stride(0)
        B_Start_Loc_stride = B_Start_Loc.stride(0)
        B_Seqlen_stride = B_Seqlen.stride(0)

        @triton.jit
        def _fwd_kernel(out_ptr, logits_ptr, v_ptr, B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr, 
                        B_Loc_stride, B_Start_Loc_stride, B_Seqlen_stride, 
                        logits_batch, logits_head, logits_start_n,
                        v_seq_len, v_head_dim,
                        ignore_index: tl.constexpr,
                        block_dmodel: tl.constexpr,
                        BLOCK_N: tl.constexpr):
            cur_batch = tl.program_id(0)
            cur_head = tl.program_id(1)
            start_n = tl.program_id(2)
            cur_block_token_num = tl.minimum(B_Seqlen[cur_batch] - start_n, BLOCK_N)
            cur_block_end = start_n + cur_block_token_num
            logits_offset = cur_batch * logits_batch + cur_head * logits_head + start_n * logits_start_n
            v_offset = cur_head * v_head_dim
            cur_batch_in_all_start_index = tl.load(B_Start_Loc_ptr + cur_batch * B_Start_Loc_stride)
            cur_batch_loc_index = B_Loc_ptr + cur_batch_in_all_start_index + start_n * B_Loc_stride

            acc = tl.zeros([cur_block_token_num, block_dmodel], dtype=tl.float32)
            max_logic = tl.zeros([cur_block_token_num], dtype=tl.float32) - float('inf')
            sum_logic = tl.zeros([cur_block_token_num], dtype=tl.float32)
            for n in range(0, cur_block_end, BLOCK_N):
                cur_n = n // BLOCK_N
                logits_ptrs = logits_ptr + logits_offset + cur_n * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = (cur_n * BLOCK_N + tl.arange(0, BLOCK_N)) < cur_block_token_num
                cur_logits = tl.load(logits_ptrs, mask=mask).to(tl.float32)
                cur_loc = tl.load(cur_batch_loc_index + cur_n * B_Loc_stride, mask=mask)
                v_ptrs = v_ptr + v_offset + cur_loc[:, None] * v_seq_len
                cur_v = tl.load(v_ptrs, mask=mask).to(tl.float32)
                cur_logic = cur_logits - tl.max(cur_logits, axis=0)
                if ignore_index != -100:
                    cur_logic = tl.where(cur_n * BLOCK_N + tl.arange(0, BLOCK_N) < B_Seqlen[cur_batch], cur_logic, ignore_index)
                max_logic = tl.maximum(max_logic, cur_logic)
                cur_logic = cur_logic - max_logic
                tl.store(cur_batch_loc_index + cur_n * B_Loc_stride, cur_logic, mask=mask)

                cur_logic = tl.exp(cur_logic)
                acc += cur_logic[:, None] * cur_v
                sum_logic += cur_logic

            logits_offset = cur_batch * logits_batch + cur_head * logits_head + cur_block_end * logits_start_n
            logits_ptrs = logits_ptr + logits_offset
            mask = cur_block_end * BLOCK_N + tl.arange(0, BLOCK_N) < seq_len * BLOCK_N
            cur_logits = tl.load(logits_ptrs, mask=mask).to(tl.float32)
            max_logic = tl.maximum(max_logic, tl.max(cur_logits, axis=0))
            cur_logits = cur_logits - max_logic
            if ignore_index != -100:
                cur_logits = tl.where(cur_block_end * BLOCK_N + tl.arange(0, BLOCK_N) < seq_len * BLOCK_N, cur_logits, ignore_index)
            cur_logits = tl.exp(cur_logits)
            acc += cur_logits[:, None] * cur_v
            sum_logic += cur_logits

            cur_block_end = tl.minimum(cur_block_end * BLOCK_N, seq_len * BLOCK_N)
            sum_logic = tl.sum(tl.where(tl.arange(0, cur_block_end) < seq_len * BLOCK_N, cur_logic, 0), axis=0)
            acc = acc / sum_logic
            out_offset = cur_batch * out_strides[0] + cur_head * out_strides[1] + start_n * out_strides[2]
            out_ptrs = out_ptr + out_offset + tl.arange(0, BLOCK_N)
            tl.store(out_ptrs, acc, mask=(tl.arange(0, BLOCK_N) < cur_block_token_num))

        grid = (batch, n_heads, triton.cdiv(seq_len, BLOCK_N))
        _fwd_kernel[grid](
            logits,
            logits,
            v,
            B_Loc,
            B_Start_Loc,
            B_Seqlen,
            B_Loc_stride,
            B_Start_Loc_stride,
            B_Seqlen_stride,
            *logits.stride(),
            v_seq_len=v_seq_len,
            v_head_dim=d_model,
            ignore_index=ignore_index,
            block_dmodel=d_model,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=1,
        )
        return logits

token_softmax_reducev_fwd = _token_softmax_reducev_fwd.apply
