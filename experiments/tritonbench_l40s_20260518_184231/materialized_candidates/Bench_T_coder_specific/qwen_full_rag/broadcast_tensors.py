import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    in0,
    in1,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    in0 = in0 + pid_x * M * N + pid_y * BLOCK_SIZE
    out = out + pid_x * M * N + pid_y * BLOCK_SIZE
    for i in range(0, BLOCK_SIZE):
        tmp = tl.load(in0 + i)
        tl.store(out + i, tmp)


def broadcast_tensors(*tensors):
    output = []
    for i in range(len(tensors)):
        x = tensors[i]
        shape_list = list(x.shape)
        for j in range(i + 1, len(tensors)):
            y = tensors[j]
            shape_list_new = []
            flag = False
            for p in range(max(len(shape_list), len(y.shape))):
                if p >= len(shape_list):
                    shape_list_new.append(y.shape[p])
                elif p >= len(y.shape):
                    shape_list_new.append(shape_list[p])
                else:
                    if shape_list[p] == y.shape[p]:
                        shape_list_new.append(shape_list[p])
                    elif shape_list[p] == 1:
                        shape_list_new.append(y.shape[p])
                        flag = True
                    elif y.shape[p] == 1:
                        shape_list_new.append(shape_list[p])
                    else:
                        raise RuntimeError("Shape mismatch")
            if flag:
                s = str(shape_list_new)
                s = s.replace(" ", "")
                print(
                    "Warning:some inputs are broadcasted!The final shapeis"
                    + s
                )
            shape_list = shape_list_new
        output.append(torch.broadcast_to(x, tuple(shape_list)))
    return output
