cpp
#include <triton/api.h>

__device__
void fused_recurrent_hgrn_fwd_kernel(
    float* x, float* g, float* o, float* h0, float* ht,
    int T, int D, int BD, bool USE_INITIAL_STATE, bool STORE_FINAL_STATE) {

    int tid = triton::program::uid::tid();
    float b_h[BD];
    float b_x[BD];
    float b_g[BD];

    if (USE_INITIAL_STATE) {
        b_h = h0;
    } else {
        for (int i = 0; i < BD; i++) {
            b_h[i] = 0.0f;
        }
    }

    for (int t = 0; t < T; t++) {
        b_x = x + t * D;
        b_g = g + t * D;

        for (int i = 0; i < BD; i++) {
            b_h[i] = b_g[i] * b_h[i] + b_x[i];
        }

        if (STORE_FINAL_STATE) {
            for (int i = 0; i < BD; i++) {
                ht[t * BD + i] = b_h[i];
            }
        }

        if (tid == 0) {
            for (int i = 0; i < BD; i++) {
                o[t * BD + i] = b_h[i];
            }
        }
    }
}

__device__
void fused_recurrent_hgrn_bwd_kernel(
    float* do_, float* dx, float* dg, float* ht,
    int T, int D, int BD, bool STORE_FINAL_STATE) {

    int tid = triton::program::uid::tid();
    float b_dh[BD];
    float b_dx[BD];
    float b_dg[BD];

    if (STORE_FINAL_STATE) {
        b_dh = ht + (T - 1) * BD;
    } else {
        for (int i = 0; i < BD; i++) {
            b_dh[i] = 0.0f;
        }
    }

    for (int t = T - 1; t >= 0; t--) {
        b_dx = dx + t * D;
        b_dg = dg + t * D;

        for (int i = 0; i < BD; i++) {
            b_dh[i] = b_dh[i] + do_[t * BD + i];
            b_dx[i] = b_dh[i];
            b_dg[i] = b_dh[i] * b_dh[i];
        }

        if (t > 0) {
            b_dh = ht + (t - 1) * BD;
        } else {
            for (int i = 0; i < BD; i++) {
                b_dh[i] = 0.0f;
            }
        }
    }
}

struct FusedRecurrentHGRNFunction : public torch::autograd::Function<FusedRecurrentHGRNFunction> {
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::Variable x, torch::autograd::Variable g,
        torch::autograd::Variable h0, int T, int D, int BD,
        bool USE_INITIAL_STATE, bool STORE_FINAL_STATE) {

        auto o = torch::zeros_like(x);
        auto ht = torch::zeros_like(h0);

        fused_recurrent_hgrn_fwd_kernel(
            x.data<float>(), g.data<float>(),
            o.data<float>(), h0.data<float>(), ht.data<float>(),
            T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE);

        ctx->save_for_backward({x, g, ht});

        return {o, ht};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {

        auto do_ = grad_outputs[0];
        auto ht = ctx->get_saved_variables()[2];

        auto dx = torch::zeros_like(do_);
        auto dg = torch::zeros_like(do_);

        fused_recurrent_hgrn_bwd_kernel(
            do_.data<float>(), dx.data<float>(), dg.data<float>(), ht.data<float>(),
            T, D, BD, STORE_FINAL_STATE);

        return {dx, dg, torch::autograd::Variable()};
    }
};

torch::autograd::Variable fused_recurrent_hgrn(
    torch::autograd::Variable x, torch::autograd::Variable g,
    torch::autograd::Variable h0, int T, int D, int BD,
    bool USE_INITIAL_STATE, bool STORE_FINAL_STATE) {

    auto outputs = FusedRecurrentHGRNFunction::apply(x, g, h0, T, D, BD, USE_INITIAL_STATE, STORE_FINAL_STATE);
    return outputs[0];
}
