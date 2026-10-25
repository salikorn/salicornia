// unit: fp64 decomposition kit (eigen recovery, exact fold, exact-rank lossless, orthonormalize)
/// \file
/// \brief Tests for the fp64 decomposition kit in compress/linalg.hpp.
///
/// Each test builds a matrix with a KNOWN spectrum from random orthogonal factors and
/// a chosen singular-value vector, so the pass/fail bounds are absolute-precision
/// claims, not cross-implementation comparisons: Jacobi eigen recovery to 1e-9 relative,
/// the square-Q SliceGPT fold to 1e-12, exact-rank truncation to 1e-12, and
/// orthonormality to 1e-10. The top-k subspace solver is only bounded loosely (3e-2)
/// because clustered spectra converge slowly — the exact-solver test carries the
/// precision claim (comment at tests/test_linalg.cpp:46).
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "compress/linalg.hpp"
#include "xlab/test.hpp"

namespace la = sal::la;

XLAB_TEST("known-spectrum eigen recovery (Jacobi + topk)") {
    const size_t n = 128, r = 96;
    std::mt19937 rng(3);
    std::normal_distribution<double> g(0, 1);
    Eigen::MatrixXd M = Eigen::MatrixXd::NullaryExpr(n, n, [&](int, int) { return g(rng); });
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();
    std::vector<double> sigma(n, 0.0);
    for (size_t i = 0; i < r; ++i) sigma[i] = std::pow(0.6, double(i) / 16.0);
    Eigen::MatrixXd W = Q * Eigen::DiagonalMatrix<double, Eigen::Dynamic>(
                                Eigen::Map<Eigen::VectorXd>(sigma.data(), n))
                                .toDenseMatrix()
                                .transpose();
    // W [n,n] with singular values sigma; G = W^T W has eigenvalues sigma^2
    std::vector<double> G(n * n);
    la::gram_f32(nullptr, 0, 0, G.data());  // no-op guard path (m=0)
    Eigen::MatrixXd Gm = W.transpose() * W;
    std::memcpy(G.data(), Gm.data(), n * n * sizeof(double));
    std::vector<double> vals, vecs;
    la::jacobi_eigen(G.data(), n, vals, vecs);
    double max_rel = 0;
    for (size_t i = 0; i < n; ++i) {
        double want = sigma[i] * sigma[i];
        if (want > 1e-24) max_rel = std::max(max_rel, std::fabs(vals[i] - want) / want);
    }
    CHECK(max_rel < 1e-9);
    CHECK(vals[r] < 1e-20);  // exactly-zero tail
    auto topk = la::topk_eigen(G.data(), n, 64, 80, 1e-13);
    double max_rel2 = 0;
    for (size_t i = 0; i < 64; ++i)
        max_rel2 = std::max(max_rel2,
                            std::fabs(topk.values[i] - sigma[i] * sigma[i]) / (sigma[i] * sigma[i]));
    // subspace iteration converges slowly on clustered spectra (adjacent ratio ~0.97 here);
    // the exact-solver check above carries the precision claim, this one bounds the heuristic
    CHECK(max_rel2 < 3e-2);
}

XLAB_TEST("square-Q exact fold (SliceGPT theorem)") {
    const size_t d = 128, m = 64;
    std::mt19937 rng(5);
    std::normal_distribution<double> g(0, 1);
    std::vector<double> x(d), gam(d), W(m * d), Q(d * d);
    for (auto& v : x) v = g(rng);
    for (auto& v : gam) v = 0.5 + std::fabs(g(rng));
    for (auto& v : W) v = g(rng);
    REQUIRE(la::random_orthogonal(d, Q.data(), 11));
    auto norm_linear = [&]() {
        double ms = 0;
        for (size_t i = 0; i < d; ++i) ms += x[i] * x[i];
        double rms = std::sqrt(ms / d);
        std::vector<double> y(m, 0.0);
        for (size_t r = 0; r < m; ++r)
            for (size_t c = 0; c < d; ++c) y[r] += W[r * d + c] * (x[c] * gam[c] / rms);
        return y;
    };
    std::vector<double> y = norm_linear();
    // x~ = Q^T x ; W~ = W diag(gam) Q
    std::vector<double> xt(d, 0.0), Wf(m * d, 0.0);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) xt[i] += Q[j * d + i] * x[j];
    for (size_t r = 0; r < m; ++r)
        for (size_t c = 0; c < d; ++c) {
            double s = 0;
            for (size_t t = 0; t < d; ++t) s += W[r * d + t] * gam[t] * Q[t * d + c];
            Wf[r * d + c] = s;
        }
    double ms2 = 0;
    for (size_t i = 0; i < d; ++i) ms2 += xt[i] * xt[i];
    double rms2 = std::sqrt(ms2 / d);  // == rms (orthogonal invariance)
    std::vector<double> yt(m, 0.0);
    for (size_t r = 0; r < m; ++r)
        for (size_t c = 0; c < d; ++c) yt[r] += Wf[r * d + c] * (xt[c] / rms2);
    double rel = std::sqrt(la::fro_sq(y.data(), yt.data(), m)) /
                 std::sqrt([&] { double s = 0; for (double v : y) s += v * v; return s; }());
    CHECK(rel < 1e-12);
}

XLAB_TEST("exact-rank truncation is lossless") {
    const size_t rank = 24, m = 96, n = 64;
    std::mt19937 rng(9);
    std::normal_distribution<double> g(0, 1);
    Eigen::MatrixXd A(m, rank), B(rank, n);
    for (auto& v : A.reshaped()) v = g(rng);
    for (auto& v : B.reshaped()) v = g(rng);
    Eigen::MatrixXd W = A * B;
    Eigen::MatrixXd G = W.transpose() * W;
    std::vector<double> Gv(n * n);
    std::memcpy(Gv.data(), G.data(), n * n * sizeof(double));
    auto eig = la::topk_eigen(Gv.data(), n, rank, 100, 1e-13);
    Eigen::MatrixXd Vr(n, rank);
    for (int i = 0; i < int(n); ++i)
        for (int j = 0; j < int(rank); ++j) Vr(i, j) = eig.vectors[size_t(i) * rank + size_t(j)];
    Eigen::MatrixXd Wr = (W * Vr) * Vr.transpose();
    double rel = (Wr - W).norm() / W.norm();
    CHECK(rel < 1e-12);
}

XLAB_TEST("orthonormalize produces Q^T Q = I") {
    const size_t n = 256, k = 64;
    std::mt19937 rng(13);
    std::normal_distribution<double> g(0, 1);
    Eigen::MatrixXd V = Eigen::MatrixXd::NullaryExpr(n, k, [&](int, int) { return g(rng); });
    REQUIRE(la::orthonormalize(V.data(), n, k));
    Eigen::MatrixXd I = V.transpose() * V;
    double err = (I - Eigen::MatrixXd::Identity(k, k)).norm();
    CHECK(err < 1e-10);
    // idempotent: a second pass does not move the basis far
    Eigen::MatrixXd V2 = V;
    REQUIRE(la::orthonormalize(V2.data(), n, k));
    CHECK((V2 - V).norm() < 1e-8);
}

/// \brief Entry point: runs the "linalg" suite (JSONL events under runs/).
/// \return 0 iff all tests passed (CTest-compatible).
int main() { return xlab::test::run_all("linalg"); }
