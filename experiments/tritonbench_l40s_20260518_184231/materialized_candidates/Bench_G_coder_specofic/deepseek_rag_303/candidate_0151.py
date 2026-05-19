import torch
import triton
import triton.language as tl


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_ROWSCALE": lambda args: args["ROWSCALE"] is not None})
@triton.heuristics({"HAS_DROPOUT": lambda args: args["DROPOUT_MASK"] is not None})
@triton.heuristics({"HAS_DROPOUT_SEED": lambda args: args["SEEDS"] is not None})
@triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.heuristics({"HAS_ADDITIONAL_INPUTS": lambda args: (args["W1"] is not None) and (args["B1"] is not None)})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
		X,  # pointer to the input
		Y,  # pointer to the output
		W,  # pointer to the weights
		B,  # pointer to the biases
		Z,  # pointer to the other branch
		RESIDUAL,  # additional residual input
		W1,  # additional linear layer weights
		B1,  # additional linear layer weights
		X1,  # additional input tensor
		Y1,  # additional input tensor
		MEAN,  # pointer to the mean
		RSTD,  # pointer to the 1/std
		DROPOUT_MASK,  # pointer to the dropout mask
		SEEDS,  # pointer to the dropout seeds
		ROWSCALE,  # rescaling input rows
		stride_x_row,  # how much to increase the pointer when moving by 1 row
		stride_y_row,
		stride_res_row,
		stride_y1_row,
		stride_z_row,
		M,  # number of rows in X
		N,  # number of columns in X
		eps,  # epsilon to avoid division by zero
		rows_per_program,
		BLOCK_N: tl.constexpr,
		HAS_BIAS: tl.constexpr,
		HAS_ROWSCALE: tl.constexpr,
		HAS_DROPOUT: tl.constexpr,
		HAS_DROPOUT_SEED: tl.constexpr,
		HAS_RESIDUAL: tl.constexpr,
		HAS_ADDITIONAL_INPUTS: tl.constexpr,
		HAS_Z: tl.constexpr,
		KEY_ANTI_CORRELATION: tl.constexpr,
		NORM_BEFORE_GATE: tl.constexpr,
		IS_RMS_NORM: tl.constexpr, MARGIN: tl.constexpr,
):
	# Map the program id to the row of X and Y it should compute.
	row_block_id = tl.program_id(0)
	group = tl.program_id(1)
	row_start = row_block_id * rows_per_program
	cols = tl.arange(0, BLOCK_N)
	mask = cols < N
	X += row_start * stride_x_row + group * N
	Y += row_start * stride_y_row + group * N
	if HAS_RESIDUAL:
		RESIDUAL += row_start * stride_res_row + group * N
	if HAS_ADDITIONAL_INPUTS:
		X1 += row_start * stride_x_row + group * N
		Y1 += row_start * stride_y1_row + group * N
	if HAS_Z:
		Z += row_start * stride_z_row + group * N
	if not IS_RMS_NORM:
		MEAN += group * M
	RSTD += group * M
	W += group * N
	if HAS_BIAS:
		B += group * N
	if HAS_ADDITIONAL_INPUTS:
		W1 += group * N
		if HAS_BIAS:
			B1 += group * N
	if HAS_DROPOUT:
		DROPOUT_MASK += row_start * N + cols
	if HAS_DROPOUT_SEED:
		SEEDS += row_block_id * M + row_start
	# Compute mean and variance
	if KEY_ANTI_CORRELATION:
		# We implement the anti-correlation dropout approximation proposed in the adaptive group norm paper
		rng0 = SEEDS * 0
		rng1 = rng0 + 1
		rng2 = rng0 + 2
		ANTI_MASK = tl.rand(rng0, rng1, rng2, row_start, (row_start + rows_per_program,))[0, 0, 0, :rows_per_program] > .5
	else:
		ANTI_MASK = tl.full([rows_per_program], 1, tl.int1)
	for _ in range(3):
		ANTI_MASK = tl.maximum(ANTI_MASK[:, None] & (ANTI_MASK[None, :]), row_start + cols[None, :] != row_start + cols[:, None])
	ANTI_MASK = tl.where(row_start + cols[:, None] == row_start + cols[None, :], 1, 0)
	x = tl.load(X + cols, mask=ANTI_MASK * mask, other=0.).to(tl.float32)
	if HAS_ROWSCALE:
		ROWSCALE += group * N
		rowscale = tl.load(ROWSCALE + cols)
		x *= rowscale[None, :]
	if HAS_RESIDUAL:
		res = tl.load(RESIDUAL + cols, mask=ANTI_MASK * mask, other=0.).to(tl.float32)
		x += res * rowscale[None, :] if HAS_ROWSCALE else res
	if HAS_Z:
		z = tl.load(Z + cols, mask=ANTI_MASK * mask, other=0.).to(tl.float32)
		x *= z * tl.sigmoid(z) if not NORM_BEFORE_GATE else 1.
	if HAS_ADDITIONAL_INPUTS:
		x1 = tl.load(X1 + cols, mask=ANTI_MASK * mask, other=0.).to(tl.float32)
		x += x1
	# regular dropout if specified
	if HAS_DROPOUT:
		dropout_mask = tl.rand(SEEDS, row_start, (row_start + rows_per_program,))[0, 0, :rows_per_program] >= MARGIN
		tl.store(DROPOUT_MASK + row_block_id * N + cols, dropout_mask)
		x = tl.where(ANTI_MASK * dropout_mask, x / MARGIN, 0.)
	# apply linear transformation and bias
	w = tl.load(W + cols, mask=mask).to(tl.float32)
	if HAS_BIAS:
		b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
	x_hat = tl.where(ANTI_MASK, (x - tl.sum(x, axis=0) / N) * rstd if not IS_RMS_NORM else x * rstd, 0.)
	y = x_hat * w + b if HAS_BIAS else x_hat * w
	if HAS_ADDITIONAL_IN
