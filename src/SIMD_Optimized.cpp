#include <immintrin.h>
#include "include/Matrix.hpp"
#include <cmath>
#include <tuple>
#include "include/Matrix.hpp"
#include "include/SIMD_Optimized.hpp" 

// SIMD 优化的矩阵乘法
Matrix matrix_multiply_simd(const Matrix &A, const Matrix &B) {
    if (A.ncol() != B.nrow()) {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }

    size_t m = A.nrow();
    size_t n = B.ncol();
    size_t k = A.ncol();

    // 預先轉置 B 矩陣
    Matrix B_transposed = B.transpose();
    Matrix result(m, n, 0.0);

    // SIMD 乘法
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            __m256d sum = _mm256_setzero_pd(); // 初始化 SIMD 累積器
            size_t kk;
            for (kk = 0; kk + 4 <= k; kk += 4) {
                // 加載 A 的一行和 B 的轉置一行
                __m256d a = _mm256_set_pd(A(i, kk + 3), A(i, kk + 2), A(i, kk + 1), A(i, kk));
                __m256d b = _mm256_loadu_pd(&B_transposed(j, kk)); // 加載轉置後的 B 行

                // FMA 操作
                sum = _mm256_fmadd_pd(a, b, sum);
            }

            // 將 SIMD 累積器中的結果存入
            double buffer[4];
            _mm256_storeu_pd(buffer, sum);
            result(i, j) = buffer[0] + buffer[1] + buffer[2] + buffer[3];

            // 處理剩餘元素
            for (; kk < k; ++kk) {
                result(i, j) += A(i, kk) * B(kk, j);
            }
        }
    }

    return result;
}

// // Jacobi 方法的 SIMD 優化
// std::tuple<Matrix, Matrix, Matrix> jacobi_eigen_simd(const Matrix &m) {
//     if (m.nrow() != m.ncol()) {
//         throw std::invalid_argument("Jacobi SVD requires a square matrix.");
//     }

//     const size_t n = m.nrow();
//     Matrix U(n, n, 0.0);
//     Matrix V(n, n, 0.0);
//     Matrix S = m; // Copy input matrix

//     const double epsilon = 1e-10; // Convergence threshold
//     const size_t max_iter = 1000000;

//     // Initialize U and V as identity matrices
//     for (size_t i = 0; i < n; ++i) {
//         U(i, i) = 1.0;
//         V(i, i) = 1.0;
//     }

//     for (size_t iter = 0; iter < max_iter; ++iter) {
//         double max_offdiag = 0.0;
//         size_t p = 0, q = 0;

//         // Find the largest off-diagonal absolute value
//         for (size_t i = 0; i < n; ++i) {
//             for (size_t j = i + 1; j < n; ++j) {
//                 if (std::abs(S(i, j)) > max_offdiag) {
//                     max_offdiag = std::abs(S(i, j));
//                     p = i;
//                     q = j;
//                 }
//             }
//         }

//         if (max_offdiag < epsilon) {
//             break; // Converged
//         }

//         // Compute rotation angle
//         double tau = (S(q, q) - S(p, p)) / (2.0 * S(p, q));
//         double t = (tau > 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
//         double cos_phi = 1.0 / std::sqrt(1.0 + t * t);
//         double sin_phi = t * cos_phi;

//         // SIMD-friendly variables
//         __m256d cos_vec = _mm256_set1_pd(cos_phi);
//         __m256d sin_vec = _mm256_set1_pd(sin_phi);

//         // Update S using SIMD
//         for (size_t i = 0; i < n; i += 4) {
//             __m256d s_ip = _mm256_loadu_pd(&S(i, p));
//             __m256d s_iq = _mm256_loadu_pd(&S(i, q));

//             __m256d sp_new = _mm256_sub_pd(_mm256_mul_pd(cos_vec, s_ip), _mm256_mul_pd(sin_vec, s_iq));
//             __m256d sq_new = _mm256_add_pd(_mm256_mul_pd(sin_vec, s_ip), _mm256_mul_pd(cos_vec, s_iq));

//             _mm256_storeu_pd(&S(i, p), sp_new);
//             _mm256_storeu_pd(&S(i, q), sq_new);
//         }

//         for (size_t j = 0; j < n; j += 4) {
//             __m256d s_pj = _mm256_loadu_pd(&S(p, j));
//             __m256d s_qj = _mm256_loadu_pd(&S(q, j));

//             __m256d sp_new = _mm256_sub_pd(_mm256_mul_pd(cos_vec, s_pj), _mm256_mul_pd(sin_vec, s_qj));
//             __m256d sq_new = _mm256_add_pd(_mm256_mul_pd(sin_vec, s_pj), _mm256_mul_pd(cos_vec, s_qj));

//             _mm256_storeu_pd(&S(p, j), sp_new);
//             _mm256_storeu_pd(&S(q, j), sq_new);
//         }

//         // Update U using SIMD
//         for (size_t i = 0; i < n; i += 4) {
//             __m256d u_ip = _mm256_loadu_pd(&U(i, p));
//             __m256d u_iq = _mm256_loadu_pd(&U(i, q));

//             __m256d up_new = _mm256_sub_pd(_mm256_mul_pd(cos_vec, u_ip), _mm256_mul_pd(sin_vec, u_iq));
//             __m256d uq_new = _mm256_add_pd(_mm256_mul_pd(sin_vec, u_ip), _mm256_mul_pd(cos_vec, u_iq));

//             _mm256_storeu_pd(&U(i, p), up_new);
//             _mm256_storeu_pd(&U(i, q), uq_new);
//         }

//         // Update V using SIMD
//         for (size_t i = 0; i < n; i += 4) {
//             __m256d v_ip = _mm256_loadu_pd(&V(i, p));
//             __m256d v_iq = _mm256_loadu_pd(&V(i, q));

//             __m256d vp_new = _mm256_sub_pd(_mm256_mul_pd(cos_vec, v_ip), _mm256_mul_pd(sin_vec, v_iq));
//             __m256d vq_new = _mm256_add_pd(_mm256_mul_pd(sin_vec, v_ip), _mm256_mul_pd(cos_vec, v_iq));

//             _mm256_storeu_pd(&V(i, p), vp_new);
//             _mm256_storeu_pd(&V(i, q), vq_new);
//         }
//     }

//     // Sort singular values and adjust U and V accordingly (same as original implementation)
//     std::vector<std::pair<double, size_t>> singular_values;
//     for (size_t i = 0; i < n; ++i) {
//         singular_values.emplace_back(S(i, i), i);
//     }
//     std::sort(singular_values.rbegin(), singular_values.rend()); // Sort in descending order

//     Matrix S_sorted(n, n, 0.0);
//     Matrix U_sorted(n, n, 0.0);
//     Matrix V_sorted(n, n, 0.0);

//     for (size_t i = 0; i < n; ++i) {
//         size_t idx = singular_values[i].second;
//         S_sorted(i, i) = singular_values[i].first;

//         for (size_t j = 0; j < n; ++j) {
//             U_sorted(j, i) = U(j, idx);
//             V_sorted(j, i) = V(j, idx);
//         }
//     }

//     return std::make_tuple(U_sorted, S_sorted, V_sorted);
// }



std::tuple<Matrix, Matrix, Matrix> svd_jacobi_simd(const Matrix &A) {
    // 確保 A 為 m x n 矩陣
    size_t m = A.nrow();
    size_t n = A.ncol();

    // 計算 A^T A (n x n)
    Matrix ATA = matrix_multiply_simd(A.transpose(), A);

    // 初始化 V 為單位矩陣 (n x n)
    Matrix V(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        V(i, i) = 1.0;
    }

    // Jacobi 特徵值分解 ATA
    const double epsilon = 1e-12;
    const size_t max_iter = 1000;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // 找最大非對角元素
        double max_offdiag = 0.0;
        size_t p = 0, q = 0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double val = std::fabs(ATA(i, j));
                if (val > max_offdiag) {
                    max_offdiag = val;
                    p = i; q = j;
                }
            }
        }

        if (max_offdiag < epsilon) {
            // 收斂
            break;
        }

        double app = ATA(p, p);
        double aqq = ATA(q, q);
        double apq = ATA(p, q);

        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0.0) ? 1.0 / (std::fabs(tau) + std::sqrt(1.0 + tau*tau))
                                : -1.0 / (std::fabs(tau) + std::sqrt(1.0 + tau*tau));
        double c = 1.0 / std::sqrt(1.0 + t*t);
        double s = t * c;

        // R 為旋轉矩陣，更新 ATA = R^T * ATA * R
        // 先更新 ATA 的行方向: 對 i, 更新 ATA(i,p) 及 ATA(i,q)
        for (size_t i = 0; i < n; ++i) {
            double ip = ATA(i, p);
            double iq = ATA(i, q);
            ATA(i, p) = c*ip + s*iq;
            ATA(i, q) = -s*ip + c*iq;
        }

        // 再更新 ATA 的列方向: 對 j, 更新 ATA(p,j) 及 ATA(q,j)
        for (size_t j = 0; j < n; ++j) {
            double pj = ATA(p, j);
            double qj = ATA(q, j);
            ATA(p, j) = c*pj + s*qj;
            ATA(q, j) = -s*pj + c*qj;
        }

        // ATA 已更新完成 (保持對稱)

        // 更新 V = V * R
        for (size_t i = 0; i < n; ++i) {
            double ip = V(i, p);
            double iq = V(i, q);
            V(i, p) = c*ip + s*iq;
            V(i, q) = -s*ip + c*iq;
        }
    }

    // 將對角線的特徵值開根號作為奇異值
    Matrix S(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double val = ATA(i, i);
        S(i, i) = (val > 0.0) ? std::sqrt(val) : 0.0;
    }

    // 計算 U = A * V * S^-1
    Matrix Sigma_inv(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (S(i, i) > 1e-15) {
            Sigma_inv(i, i) = 1.0 / S(i, i);
        }
    }

    Matrix U = matrix_multiply_simd(A, matrix_multiply_simd(V, Sigma_inv));

    // 若 A 為 m x n, U 為 m x n, S 為 n x n, V 為 n x n
    // 若需要完整的 m x m U 矩陣（若 m > n），可再填補。
    // 這裡假設 m >= n，若 m < n，SVD 的形狀可能需要調整。

    return {U, S, V};
}