/**
 * \file linalg.cpp
 * \brief Eigen-backed implementations for the declarations in linalg.hpp.
 *
 * Compiled with /openmp (CMake) so Eigen's OpenMP-parallel products behave
 * exactly as when these bodies were inlined into the /openmp consumers.
 */
#include "compress/linalg.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "xlab/fp16.hpp"

namespace sal::la {

unsigned nthreads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 8;
}

void gemm(const double* A, const double* B, double* C, size_t M, size_t K, size_t N,
          double alpha, double beta, unsigned /*nt*/) {
    using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const Mat> a(A, Eigen::Index(M), Eigen::Index(K));
    Eigen::Map<const Mat> b(B, Eigen::Index(K), Eigen::Index(N));
    Eigen::Map<Mat> c(C, Eigen::Index(M), Eigen::Index(N));
    if (beta == 0.0)
        c.noalias() = alpha * a * b;
    else
        c = beta * c + alpha * a * b;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> load_matrix_bf16(
    const uint16_t* X, size_t m, size_t n) {
    using RMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RMat Xd(m, n);
    parallel_for(m, nthreads(), [&](size_t r) {
        double* row = Xd.data() + r * n;
        const uint16_t* src = X + r * n;
        for (size_t c = 0; c < n; ++c) row[c] = double(sal::bf16_to_f32(src[c]));
    });
    return Xd;
}

void gram_bf16(const uint16_t* X, size_t m, size_t n, double* G, unsigned /*nt*/,
               bool transposed) {
    using RMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RMat Xd(m, n);
    parallel_for(m, nthreads(), [&](size_t r) {
        double* row = Xd.data() + r * n;
        if (!transposed) {
            const uint16_t* src = X + r * n;
            for (size_t c = 0; c < n; ++c) row[c] = double(sal::bf16_to_f32(src[c]));
        } else {
            for (size_t c = 0; c < n; ++c) row[c] = double(sal::bf16_to_f32(X[c * m + r]));
        }
    });
    Eigen::Map<Eigen::MatrixXd> g(G, Eigen::Index(n), Eigen::Index(n));
    g.noalias() = Xd.transpose() * Xd;
}

void gram_f32(const float* X, size_t m, size_t n, double* G, unsigned /*nt*/) {
    using RMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RMat Xd(m, n);
    parallel_for(m, nthreads(), [&](size_t r) {
        double* row = Xd.data() + r * n;
        const float* src = X + r * n;
        for (size_t c = 0; c < n; ++c) row[c] = double(src[c]);
    });
    Eigen::Map<Eigen::MatrixXd> g(G, Eigen::Index(n), Eigen::Index(n));
    g.noalias() = Xd.transpose() * Xd;
}

bool orthonormalize(double* V, size_t n, size_t k, unsigned /*nt*/) {
    Eigen::Map<Eigen::MatrixXd> Vm(V, Eigen::Index(n), Eigen::Index(k));
    Eigen::MatrixXd C = Vm.transpose() * Vm;
    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success) {
        C.diagonal().array() += 1e-12 * C.trace() / double(k) + 1e-300;
        llt.compute(C);
        if (llt.info() != Eigen::Success) return false;
    }
    Eigen::MatrixXd LtInv = llt.matrixU().solve(Eigen::MatrixXd::Identity(Eigen::Index(k), Eigen::Index(k)));
    Vm = Vm * LtInv;
    return true;
}

TopK topk_eigen(const double* G, size_t n, size_t k, int max_iters, double tol, unsigned nt) {
    if (k > n) k = n;
    (void)nt;
    Eigen::Map<const Eigen::MatrixXd> Gm(G, Eigen::Index(n), Eigen::Index(n));
    std::mt19937_64 rng(20261025u);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd V(n, k);
    for (auto& v : V.reshaped()) v = gauss(rng);
    if (!orthonormalize(V.data(), n, k)) throw std::runtime_error("orthonormalize(V0) failed");

    Eigen::MatrixXd W(n, k);
    Eigen::VectorXd lam(k), lamPrev = Eigen::VectorXd::Zero(k);
    bool converged = false;
    int it = 0;
    for (; it < max_iters && !converged; ++it) {
        W.noalias() = Gm * V;
        double max_res = 0.0, max_change = 0.0;
        for (Eigen::Index c = 0; c < Eigen::Index(k); ++c) {
            double l = V.col(c).dot(W.col(c));
            lam(c) = l;
            max_res = std::max(max_res, (W.col(c) - l * V.col(c)).norm());
            max_change = std::max(max_change, std::fabs(l - lamPrev(c)));
        }
        if (it > 0 && max_change < tol * std::max(1.0, lam(0))) converged = true;
        lamPrev = lam;
        if (!orthonormalize(W.data(), n, k)) throw std::runtime_error("orthonormalize failed");
        V.swap(W);
    }
    W.noalias() = Gm * V;
    std::vector<size_t> order(k);
    for (size_t i = 0; i < k; ++i) order[i] = i;
    for (Eigen::Index c = 0; c < Eigen::Index(k); ++c) lam(c) = V.col(c).dot(W.col(c));
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return lam(a) > lam(b); });
    TopK out;
    out.iters_used = it;
    out.values.assign(k, 0.0);
    out.vectors.assign(n * k, 0.0);
    for (size_t c = 0; c < k; ++c) {
        out.values[c] = std::max(0.0, lam(order[c]));
        for (size_t r = 0; r < n; ++r) out.vectors[r * k + c] = V(r, order[c]);
    }
    return out;
}

void jacobi_eigen(const double* A_in, size_t n, std::vector<double>& vals,
                  std::vector<double>& vecs) {
    Eigen::Map<const Eigen::MatrixXd> A(A_in, Eigen::Index(n), Eigen::Index(n));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    if (es.info() != Eigen::Success) throw std::runtime_error("SelfAdjointEigenSolver failed");
    vals.resize(n);
    vecs.assign(n * n, 0.0);
    for (size_t c = 0; c < n; ++c) {
        vals[c] = es.eigenvalues()(Eigen::Index(n - 1 - c));  // ascending -> descending
        for (size_t r = 0; r < n; ++r) vecs[r * n + c] = es.eigenvectors()(Eigen::Index(r), Eigen::Index(n - 1 - c));
    }
}

bool random_orthogonal(size_t n, double* Q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (auto& v : M.reshaped()) v = gauss(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Qd = qr.householderQ();
    // deterministic sign convention so it's reproducible
    for (Eigen::Index c = 0; c < Qd.cols(); ++c)
        if (Qd(0, c) < 0) Qd.col(c) *= -1.0;
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < n; ++c) Q[r * n + c] = Qd(r, c);
    return true;
}

double fro_sq(const double* a, const double* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

}  // namespace sal::la
