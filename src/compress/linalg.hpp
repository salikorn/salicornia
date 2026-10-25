/**
 * \file linalg.hpp
 * \brief fp64 CPU linear-algebra kit for decomposition experiments (Eigen-backed).
 *
 * Built on the vendored Eigen (MPL2): SIMD + OpenMP GEMM/syrk, LLT,
 * SelfAdjointEigenSolver, Householder QR. Provides subspace iteration for the
 * top-k spectrum of large Gramians (useful when k is close to n and a full
 * eigendecomposition is too slow). The API is the same one the spikes already use.
 *
 * Buffer-layout convention: all raw-pointer interfaces take/expose \e row-major
 * buffers at the boundary; internally Eigen's column-major storage is used.
 *
 * This header holds the declarations plus the header-only parallel_for template;
 * the Eigen-backed implementations live in linalg.cpp (the Eigen include moved
 * there too — this header only forward-declares Eigen::Matrix for
 * load_matrix_bf16's return type).
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

// Forward declaration matching Eigen's own (see Eigen/src/Core/util/ForwardDeclarations.h);
// defaults are supplied by Eigen's header when it is included (in linalg.cpp).
namespace Eigen {
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Matrix;
}

namespace sal::la {

/// Number of hardware threads available; falls back to 8 when
/// std::thread::hardware_concurrency() reports 0 (unknown).
unsigned nthreads();

/// \brief Thread fan-out of f over the index range [0, n).
///
/// Splits [0, n) into `nt` contiguous chunks and runs f(i) for each index on
/// its own std::thread, joining all threads before returning. Header-only:
/// template, instantiated at call sites.
///
/// \param n  Number of indices to iterate.
/// \param nt Requested thread count; when nt <= 1 or n <= 1 the loop runs
///           serially on the calling thread.
/// \param f  Callable invoked as f(size_t i) for every i in [0, n). Must not
///           throw across the thread boundary.
template <class F>
void parallel_for(size_t n, unsigned nt, F&& f) {
    if (nt <= 1 || n <= 1) {
        for (size_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::vector<std::thread> ts;
    size_t chunk = (n + nt - 1) / nt;
    ts.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
        size_t b = size_t(t) * chunk, e = std::min(n, b + chunk);
        if (b >= e) break;
        ts.emplace_back([b, e, &f] {
            for (size_t i = b; i < e; ++i) f(i);
        });
    }
    for (auto& th : ts) th.join();
}

/// \brief C[M,N] = alpha * A[M,K] * B[K,N] + beta * C, all buffers row-major.
///
/// Backed by an Eigen GEMM on mapped row-major matrices. When beta == 0 the
/// result overwrites C without reading it (so C may contain NaNs/garbage);
/// otherwise C is read and accumulated into.
///
/// \param A     Row-major M x K input matrix (M*K doubles).
/// \param B     Row-major K x N input matrix (K*N doubles).
/// \param C     Row-major M x N in/out matrix (M*N doubles).
/// \param M     Rows of A / C.
/// \param K     Columns of A / rows of B.
/// \param N     Columns of B / C.
/// \param alpha Scale on the product A*B.
/// \param beta  Scale on the existing contents of C.
///
/// The trailing unnamed thread-count parameter is accepted for API
/// compatibility and unused; Eigen's OpenMP threading drives the parallelism.
void gemm(const double* A, const double* B, double* C, size_t M, size_t K, size_t N,
          double alpha, double beta, unsigned /*nt*/ = 0);

/// \brief Convert bf16 row-major X[m,n] into an Eigen row-major double matrix.
///
/// Each bf16 element is widened via sal::bf16_to_f32, using all threads
/// (parallel_for over rows).
///
/// \param X  Row-major m x n bf16 buffer (uint16_t bit patterns).
/// \param m  Number of rows.
/// \param n  Number of columns.
/// \return An m x n row-major Eigen::Matrix<double,...> with the widened values.
///
/// \note The return type is spelled with Eigen's fixed layout constants
///       (Dynamic = -1, RowMajor = 1) so this header needs no Eigen include.
Eigen::Matrix<double, -1, -1, 1, -1, -1> load_matrix_bf16(const uint16_t* X, size_t m, size_t n);

/// \brief G[n,n] = X[m,n]^T * X[m,n] (Gram matrix); X is bf16 row-major.
///
/// Widens X to double row-by-row (parallel over rows), then computes the
/// n x n Gram product into G through an Eigen column-major map.
///
/// \param X          Row-major m x n bf16 buffer.
/// \param m          Number of rows of X.
/// \param n          Number of columns of X (and of G).
/// \param G          Output n x n row-major Gram matrix.
/// \param transposed When true, X is stored as the [n,m] transpose (smaller
///                   dimension first), i.e. logical X[i,c] = X[c*m + i].
///
/// The trailing unnamed thread-count parameter is accepted for API
/// compatibility and unused.
void gram_bf16(const uint16_t* X, size_t m, size_t n, double* G, unsigned /*nt*/ = 0,
               bool transposed = false);

/// \brief G[n,n] = X[m,n]^T * X[m,n] (Gram matrix); X is f32 row-major.
///
/// Same contract as gram_bf16() but widening from float instead of bf16.
///
/// \param X  Row-major m x n float buffer.
/// \param m  Number of rows of X.
/// \param n  Number of columns of X (and of G).
/// \param G  Output n x n row-major Gram matrix.
///
/// The trailing unnamed thread-count parameter is accepted for API
/// compatibility and unused.
void gram_f32(const float* X, size_t m, size_t n, double* G, unsigned /*nt*/ = 0);

/// \brief Orthonormalize the columns of V (n x k, stored COLUMN-major, i.e.
///        Eigen layout) in place via Cholesky QR.
///
/// Computes C = V^T V = L L^T, then replaces V by V * L^{-T}, which has
/// orthonormal columns when the factorization succeeds. If the plain Cholesky
/// fails (C numerically rank-deficient), the diagonal of C is jittered by
/// 1e-12 * trace(C)/k + 1e-300 and the factorization is retried once.
///
/// \param V  In/out n x k column-major buffer (k column vectors of length n).
/// \param n  Row count of V.
/// \param k  Column count of V.
/// \return true if V was orthonormalized (possibly after jittering); false if
///         the Cholesky failed even after the retry (the failed first attempt
///         does not write V).
///
/// The trailing unnamed thread-count parameter is accepted for API
/// compatibility and unused. Requires k <= n for C to be positive definite in
/// exact arithmetic.
bool orthonormalize(double* V, size_t n, size_t k, unsigned /*nt*/ = 0);

/// \brief Result of topk_eigen(): top-k eigenpairs of a symmetric PSD matrix.
struct TopK {
    std::vector<double> values;    ///< Eigenvalues in descending order (length k; clamped to >= 0).
    std::vector<double> vectors;   ///< n x k ROW-major buffer; column j is the eigenvector for values[j].
    int iters_used = 0;            ///< Subspace iterations actually performed (early exit on convergence).
    double max_residual = 0.0;     ///< Max over columns of ||G v - lambda v|| observed during the last iteration.
};

/// \brief Top-k eigenpairs of symmetric PSD G (n x n, row-major buffer) via
///        subspace iteration powered by Eigen products.
///
/// Starts from a seeded-Gaussian random block V (n x k, fixed seed 20261025 so
/// runs are reproducible), orthonormalized with orthonormalize(); each
/// iteration computes W = G*V, Rayleigh-quotient estimates lam(c) = v_c^T W v_c,
/// and re-orthonormalizes W. Convergence is declared after the first iteration
/// when max_c |lam(c) - lamPrev(c)| < tol * max(1, lam(0)); iteration also stops
/// after max_iters. A final W = G*V pass re-estimates the eigenvalues and the
/// columns are sorted by descending value.
///
/// \param G          Row-major n x n symmetric PSD matrix.
/// \param n          Dimension of G.
/// \param k          Number of eigenpairs wanted; clamped to n when k > n.
/// \param max_iters  Iteration cap (default 40).
/// \param tol        Relative eigenvalue-change tolerance (default 1e-11).
/// \param nt         Accepted for API compatibility; currently unused.
/// \return TopK with values descending (clamped to >= 0), vectors row-major
///         n x k with columns = eigenvectors, plus iters_used and max_residual
///         from the last iteration.
///
/// \note Throws std::runtime_error if the initial or any per-iteration
///       orthonormalize() fails (numerically singular random block).
TopK topk_eigen(const double* G, size_t n, size_t k, int max_iters = 40,
                double tol = 1e-11, unsigned nt = 0);

/// \brief Full symmetric eigensolver (Eigen SelfAdjointEigenSolver).
///
/// \param A_in  Row-major n x n symmetric input matrix (the buffer is only
///              read, not modified).
/// \param n     Dimension of A_in.
/// \param vals  Output eigenvalues in DESCENDING order (length n).
/// \param vecs  Output row-major n x n matrix whose column j is the
///              eigenvector for vals[j].
///
/// \note Throws std::runtime_error if the Eigen solver reports failure.
void jacobi_eigen(const double* A_in, size_t n, std::vector<double>& vals,
                  std::vector<double>& vecs);

/// \brief Random orthogonal n x n matrix via Householder QR of a Gaussian
///        matrix (row-major out).
///
/// Draws an n x n standard-Gaussian matrix from mt19937_64 seeded with \p seed,
/// applies Eigen's HouseholderQR, then applies a deterministic sign convention
/// (each column whose top entry is negative is negated) so the output is
/// reproducible for a given seed.
///
/// \param n    Dimension of the matrix.
/// \param Q    Output buffer, n*n doubles, row-major.
/// \param seed RNG seed (default 1).
/// \return Always true (QR of a Gaussian matrix succeeds; retained for API
///         symmetry with orthonormalize()).
bool random_orthogonal(size_t n, double* Q, uint64_t seed = 1);

/// \brief Sum of squared elementwise differences of two length-n buffers:
///        ||a - b||^2 (squared Frobenius norm for matrix data).
///
/// \param a  First buffer (n doubles).
/// \param b  Second buffer (n doubles).
/// \param n  Number of elements to compare.
/// \return sum_i (a[i] - b[i])^2.
double fro_sq(const double* a, const double* b, size_t n);

}  // namespace sal::la
