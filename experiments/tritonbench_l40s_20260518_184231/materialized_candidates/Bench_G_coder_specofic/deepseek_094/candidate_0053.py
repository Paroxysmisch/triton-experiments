# Triton Language
.version 1

.global D_MODEL_K: u32
.global D_MODEL_V: u32
.global BLOCK_MODEL_K: u32
.global BLOCK_MODEL_V: u32
.global NUM_HEAD: u32
.global NUM_BLOCK: u32

.align 16
.global last_kv: [[D_MODEL_K, D_MODEL_V]]

.align 16
.global S: [[D_MODEL_K, D_MODEL_V]]
.global d: [[D_MODEL_K, D_MODEL_V]]
.global O: [[D_MODEL_K, D_MODEL_V]]
.global last_kv: [[D_MODEL_K, D_MODEL_V]]

.align 16
.global DI: [[D_MODEL_K, D_MODEL_V]]
.global DG: [[D_MODEL_K, D_MODEL_V]]
.global DL: [[D_MODEL_K, D_MODEL_V]]
.global DS: [[D_MODEL_K, D_MODEL_V]]

.func _fwd_recurrence()
.func _bwd_recurrence()

.export _fwd_recurrence
.export _bwd_recurrence
