// salikorn / non-linear decomposition spike — does NON-LINEAR beat LINEAR at equal bits
// on real weights? (User direction: try non-linear decompositions.)
// Codec bake-off on 16x16 patches of a real shard tensor:
//   fp16 | int8 RTN | VQ k-means {16,64,256} | linear autoencoder | MLP (GELU) autoencoder
//   vs the measured global SVD distortion curve (from the fp64 spectrum of the same tensor).
// Metric: relative MSE on held-out patches vs bits-per-weight. Events -> runs/.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "compress/linalg.hpp"
#include "st/safetensors.hpp"
#include "xlab/fp16.hpp"
#include "xlab/log.hpp"

namespace la = sal::la;
using Eigen::MatrixXf;
using Eigen::VectorXf;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
static const double kPi = 3.14159265358979323846;
static float gelu(float x) {
    const float k0 = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + 0.044715f * x * x * x)));
}
static float dgelu(float x) {
    const float k0 = 0.7978845608f;
    float x3 = x + 0.044715f * x * x * x;
    float t = std::tanh(k0 * x3);
    float dt = (1 - t * t) * k0 * (1 + 3 * 0.044715f * x * x);
    return 0.5f * (1 + t) + 0.5f * x * dt;
}

struct AE {
    bool nonlinear;
    int latent;
    MatrixXf W1, W2, W3, W4;   // 384x256, Lx384, 384xL, 256x384
    VectorXf b1, b2, b3, b4;
    MatrixXf mW1, mW2, mW3, mW4, vW1, vW2, vW3, vW4;   // Adam moments
    VectorXf mb1, mb2, mb3, mb4, vb1, vb2, vb3, vb4;
    long step = 0;

    AE(int L, bool nl, std::mt19937& rng) : nonlinear(nl), latent(L) {
        std::normal_distribution<float> g(0, 1);
        auto mk = [&](int rows, int cols, int fanin) {
            MatrixXf t(rows, cols);
            float s = std::sqrt(2.0f / fanin);
            for (auto& v : t.reshaped()) v = g(rng) * s;
            return t;
        };
        W1 = mk(384, 256, 256);
        W2 = mk(L, 384, 384);
        W3 = mk(384, L, L);
        W4 = mk(256, 384, 384);
        b1.setZero(384); b2.setZero(L); b3.setZero(384); b4.setZero(256);
        mW1.setZero(384, 256); vW1.setZero(384, 256);
        mW2.setZero(L, 384); vW2.setZero(L, 384);
        mW3.setZero(384, L); vW3.setZero(384, L);
        mW4.setZero(256, 384); vW4.setZero(256, 384);
        mb1.setZero(384); mb2.setZero(L); mb3.setZero(384); mb4.setZero(256);
        vb1.setZero(384); vb2.setZero(L); vb3.setZero(384); vb4.setZero(256);
    }

    // forward batch [B x 256] -> reconstruction, keeping intermediates for backward
    struct Fwd {
        MatrixXf X, H1, Z, H3, Xh;
    };
    Fwd forward(const MatrixXf& X) const {
        Fwd f;
        f.X = X;
        MatrixXf A1 = X * W1.transpose();
        A1.rowwise() += b1.transpose();
        if (nonlinear)
            for (auto& v : A1.reshaped()) v = gelu(v);
        f.H1 = A1;
        f.Z = f.H1 * W2.transpose();
        f.Z.rowwise() += b2.transpose();
        MatrixXf A3 = f.Z * W3.transpose();
        A3.rowwise() += b3.transpose();
        if (nonlinear)
            for (auto& v : A3.reshaped()) v = gelu(v);
        f.H3 = A3;
        f.Xh = f.H3 * W4.transpose();
        f.Xh.rowwise() += b4.transpose();
        return f;
    }
    float loss(const Fwd& f) const { return (f.Xh - f.X).squaredNorm() / float(f.X.size()); }

    void train_step(const MatrixXf& X, float lr) {
        Fwd f = forward(X);
        float B = float(X.rows());
        MatrixXf D = 2.0f / float(f.X.size()) * (f.Xh - f.X);   // dL/dXh
        // backward
        MatrixXf gW4 = D.transpose() * f.H3;
        VectorXf gb4 = D.colwise().sum();
        MatrixXf DH3 = D * W4;
        if (nonlinear) {
            MatrixXf A3 = f.Z * W3.transpose();
            A3.rowwise() += b3.transpose();
            for (int r = 0; r < DH3.rows(); ++r)
                for (int c = 0; c < DH3.cols(); ++c) DH3(r, c) *= dgelu(A3(r, c));
        }
        MatrixXf gW3 = DH3.transpose() * f.Z;
        VectorXf gb3 = DH3.colwise().sum();
        MatrixXf DZ = DH3 * W3;
        MatrixXf gW2 = DZ.transpose() * f.H1;
        VectorXf gb2 = DZ.colwise().sum();
        MatrixXf DH1 = DZ * W2;
        if (nonlinear) {
            MatrixXf A1 = f.X * W1.transpose();
            A1.rowwise() += b1.transpose();
            for (int r = 0; r < DH1.rows(); ++r)
                for (int c = 0; c < DH1.cols(); ++c) DH1(r, c) *= dgelu(A1(r, c));
        }
        MatrixXf gW1 = DH1.transpose() * f.X;
        VectorXf gb1 = DH1.colwise().sum();
        (void)B;
        // Adam update on all params
        step++;
        float b1a = 0.9f, b2a = 0.999f, eps = 1e-8f;
        float bc1 = 1 - std::pow(b1a, (float)step), bc2 = 1 - std::pow(b2a, (float)step);
        auto upd = [&](auto& P, auto& M, auto& V, const auto& G) {
            const float* g = G.data();
            for (int i = 0; i < P.size(); ++i) {
                M.data()[i] = b1a * M.data()[i] + (1 - b1a) * g[i];
                V.data()[i] = b2a * V.data()[i] + (1 - b2a) * g[i] * g[i];
                P.data()[i] -= lr * (M.data()[i] / bc1) / (std::sqrt(V.data()[i] / bc2) + eps);
            }
        };
        upd(W1, mW1, vW1, gW1); upd(W2, mW2, vW2, gW2);
        upd(W3, mW3, vW3, gW3); upd(W4, mW4, vW4, gW4);
        upd(b1, mb1, vb1, gb1); upd(b2, mb2, vb2, gb2);
        upd(b3, mb3, vb3, gb3); upd(b4, mb4, vb4, gb4);
    }
};

static float rel_mse(const MatrixXf& x, const MatrixXf& xh) {
    return float((xh - x).squaredNorm() / x.squaredNorm());
}

int main(int argc, char** argv) {
    std::string shard = "models/GLM-5.3-Flash/model-00062-of-00062.safetensors";
    std::string tensor = "model.language_model.layers.9.self_attn.q_proj.weight";
    int steps = 1500, batch = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--shard") shard = argv[++i];
        else if (a == "--tensor") tensor = argv[++i];
        else if (a == "--steps") steps = std::stoi(argv[++i]);
        else if (a == "--batch") batch = std::stoi(argv[++i]);
    }
    xlab::Run run("nonlinear-codec");
    try {
        sal::st::SafeTensorsFile f{std::filesystem::path(shard)};
        const sal::st::TensorInfo* t = f.find(tensor);
        if (!t) throw std::runtime_error("tensor not found");
        const size_t m = size_t(t->shape[0]), n = size_t(t->shape[1]);
        const uint16_t* X = reinterpret_cast<const uint16_t*>(t->data);
        run.event("tensor", "\"name\":\"" + tensor + "\",\"m\":" + std::to_string(m) +
                                ",\"n\":" + std::to_string(n));

        // patches 16x16 -> rows; deterministic split
        const int P16 = 16;
        const size_t pr = m / P16, pc = n / P16, npatch = pr * pc;
        std::mt19937 rng(11);
        std::vector<size_t> order(npatch);
        for (size_t i = 0; i < npatch; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        size_t ntr = std::min<size_t>(40000, npatch), nva = std::min<size_t>(8000, npatch - ntr);
        auto patch_row = [&](size_t idx, MatrixXf& out, int row) {
            size_t r = (idx / pc) * P16, c = (idx % pc) * P16;
            for (int i = 0; i < P16; ++i)
                for (int j = 0; j < P16; ++j)
                    out(row, i * P16 + j) = sal::bf16_to_f32(X[(r + i) * n + c + j]);
        };
        MatrixXf Tr(ntr, 256), Va(nva, 256);
        for (size_t i = 0; i < ntr; ++i) patch_row(order[i], Tr, int(i));
        for (size_t i = 0; i < nva; ++i) patch_row(order[ntr + i], Va, int(i));

        // normalize by global scale (per-tensor)
        float scale = 0;
        for (int i = 0; i < Tr.size(); ++i) scale = std::max(scale, std::fabs(Tr.data()[i]));
        Tr.array() /= scale; Va.array() /= scale;

        auto report = [&](const char* codec, double bpw, double rmse_rel, const char* extra = "") {
            char b[256];
            std::snprintf(b, sizeof b, "\"codec\":\"%s\",\"bpw\":%.3f,\"rel_mse\":%.6e%s", codec,
                          bpw, rmse_rel, extra);
            run.event("codec_point", b);
        };

        // baselines
        report("fp16", 16.0, 0.0);
        {
            // int8 per-tensor RTN
            double err = 0, tot = 0;
            for (int i = 0; i < Va.size(); ++i) {
                float x = Va.data()[i];
                float q = std::round(std::max(-1.0f, std::min(1.0f, x)) * 127.0f) / 127.0f;
                double d = double(x) - double(q);
                err += d * d; tot += double(x) * double(x);
            }
            report("int8_rtn", 8.0, err / tot);
        }
        // VQ k-means
        for (int K : {16, 64, 256}) {
            double t0 = now_s();
            MatrixXf cb(K, 256);
            {
                std::mt19937 kr(5);
                std::uniform_int_distribution<int> pick(0, int(ntr) - 1);
                for (int k = 0; k < K; ++k) {
                    int idx = pick(kr);
                    for (int j = 0; j < 256; ++j) cb(k, j) = Tr(idx, j);
                }
            }
            MatrixXf sample = Tr.topRows(std::min<int>(int(ntr), 20000));
            for (int it = 0; it < 15; ++it) {
                // assign
                Eigen::VectorXi asg(sample.rows());
                Eigen::MatrixXf D2 = sample * cb.transpose();          // [S,K]
                Eigen::MatrixXf n2s = sample.rowwise().squaredNorm();  // [S,1]
                Eigen::MatrixXf n2c = cb.rowwise().squaredNorm().transpose();
                for (int s = 0; s < sample.rows(); ++s) {
                    float best = 1e30f; int bi = 0;
                    for (int k = 0; k < K; ++k) {
                        float d = n2s(s) - 2 * D2(s, k) + n2c(k);
                        if (d < best) { best = d; bi = k; }
                    }
                    asg(s) = bi;
                }
                // update
                Eigen::MatrixXf sums = MatrixXf::Zero(K, 256);
                Eigen::VectorXi cnt = Eigen::VectorXi::Zero(K);
                for (int s = 0; s < sample.rows(); ++s) {
                    sums.row(asg(s)) += sample.row(s);
                    cnt(asg(s))++;
                }
                for (int k = 0; k < K; ++k)
                    if (cnt(k) > 0) cb.row(k) = sums.row(k) / float(cnt(k));
            }
            // evaluate on val
            double err = 0, tot = 0;
            Eigen::MatrixXf D2 = Va * cb.transpose();
            Eigen::MatrixXf n2v = Va.rowwise().squaredNorm();
            Eigen::MatrixXf n2c = cb.rowwise().squaredNorm().transpose();
            for (int s = 0; s < Va.rows(); ++s) {
                float best = 1e30f; int bi = 0;
                for (int k = 0; k < K; ++k) {
                    float d = n2v(s) - 2 * D2(s, k) + n2c(k);
                    if (d < best) { best = d; bi = k; }
                }
                err += best; tot += n2v(s);
            }
            double bpw = std::log2(double(K)) + double(K) * 256 * 16 / double(npatch) / 256.0;
            char extra[64];
            std::snprintf(extra, sizeof extra, ",\"sec\":%.1f", now_s() - t0);
            report("vq_kmeans", bpw, err / tot, extra);
        }

        // SVD curve (CPU fp64 Gram, k=1024) -> tensor-global rel MSE = tail/trace
        {
            double t0 = now_s();
            std::vector<double> G(n * n);
            la::gram_bf16(X, m, n, G.data());
            double trace = 0;
            for (size_t i = 0; i < n; ++i) trace += G[i * n + i];
            auto eig = la::topk_eigen(G.data(), n, 1024, 20, 1e-11);
            double acc = 0;
            for (size_t i = 0; i < 1024; ++i) {
                acc += eig.values[i];
                if (i + 1 == 128 || i + 1 == 256 || i + 1 == 512 || i + 1 == 1024) {
                    size_t r = i + 1;
                    double bpw = 16.0 * double(m + n) * r / double(m) / double(n);
                    double rmse = (trace - acc) / trace;
                    report("svd_global", bpw, rmse);
                }
            }
            char extra[64];
            std::snprintf(extra, sizeof extra, ",\"sec\":%.1f", now_s() - t0);
            run.event("svd_done", std::string(extra).substr(1));
        }

        // autoencoders: linear + MLP, latent L in {32, 64, 96}
        for (int L : {32, 64, 96}) {
            for (int nl = 0; nl < 2; ++nl) {
                double t0 = now_s();
                std::mt19937 ar(7 + L + nl);
                AE ae(L, nl == 1, ar);
                std::uniform_int_distribution<int> pick(0, int(ntr) - 1);
                for (int s = 0; s < steps; ++s) {
                    float lr = 3e-3f * 0.5f * (1 + std::cos(float(kPi) * s / steps));
                    MatrixXf B(batch, 256);
                    for (int b = 0; b < batch; ++b) B.row(b) = Tr.row(pick(ar));
                    ae.train_step(B, lr);
                }
                AE::Fwd fv = ae.forward(Va);
                double rmse = rel_mse(Va, fv.Xh);
                // latent fp16 quantization penalty (round-trip latents)
                MatrixXf A1 = Va * ae.W1.transpose();
                A1.rowwise() += ae.b1.transpose();
                if (ae.nonlinear) for (auto& v : A1.reshaped()) v = gelu(v);
                MatrixXf Z = A1 * ae.W2.transpose();
                Z.rowwise() += ae.b2.transpose();
                for (auto& v : Z.reshaped())
                    v = sal::f16_to_f32(sal::f32_to_f16(v));
                MatrixXf A3 = Z * ae.W3.transpose();
                A3.rowwise() += ae.b3.transpose();
                if (ae.nonlinear) for (auto& v : A3.reshaped()) v = gelu(v);
                MatrixXf Xq = A3 * ae.W4.transpose();
                Xq.rowwise() += ae.b4.transpose();
                double rmse_q = rel_mse(Va, Xq);
                double bpw = double(L) * 16.0 / 256.0;  // fp16 latents
                double shared = (384.0 * 256 + 384.0 * L * 2 + 256.0 * 384) * 16.0 /
                                double(npatch) / 256.0;
                char extra[128];
                std::snprintf(extra, sizeof extra,
                              ",\"rel_mse_latent_fp16\":%.6e,\"sec\":%.1f", rmse_q, now_s() - t0);
                report(nl ? "mlp_ae" : "linear_ae", bpw + shared, rmse, extra);
            }
        }
        run.event("done", "\"status\":\"ok\"");
        return 0;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
