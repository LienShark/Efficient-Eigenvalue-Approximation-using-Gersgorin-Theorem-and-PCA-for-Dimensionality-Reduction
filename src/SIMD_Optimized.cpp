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

// SIMD 优化的对称矩阵 SVD (基于特征值分解)
std::tuple<Matrix, Matrix, Matrix> svd_jacobi_simd(const Matrix &A) {
    // 確保矩陣是方陣
    if (A.nrow() != A.ncol()) {
        throw std::invalid_argument("Matrix must be square for symmetric SVD.");
    }

    size_t n = A.nrow();

    // 初始化 U、S 和 V 矩陣
    Matrix U(n, n, 0.0);
    Matrix S(n, n, 0.0);
    Matrix V(n, n, 0.0);

    // 初始化 U 為單位矩陣
    for (size_t i = 0; i < n; ++i) {
        U(i, i) = 1.0;
    }

    // Step 1: 計算 A^T A
    Matrix ATA = matrix_multiply_simd(A, A);

    // Step 2: 使用 Jacobi 方法對 A^T A 進行特徵值分解
    const double epsilon = 1e-10; // 收斂閾值
    const size_t max_iter = 1000; // 最大迭代次數

    for (size_t iter = 0; iter < max_iter; ++iter) {
        double max_offdiag = 0.0;
        size_t p = 0, q = 0;

        // 找到絕對值最大的非對角元素
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(ATA(i, j)) > max_offdiag) {
                    max_offdiag = std::abs(ATA(i, j));
                    p = i;
                    q = j;
                }
            }
        }

        // 收斂條件
        if (max_offdiag < epsilon) {
            break;
        }

        // 計算旋轉角度
        double tau = (ATA(q, q) - ATA(p, p)) / (2.0 * ATA(p, q));
        double t = (tau > 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        double cos_phi = 1.0 / std::sqrt(1.0 + t * t);
        double sin_phi = t * cos_phi;

        // SIMD 旋轉矩陣更新
        __m256d cos_vec = _mm256_set1_pd(cos_phi);
        __m256d sin_vec = _mm256_set1_pd(sin_phi);

        for (size_t k = 0; k < n; k += 4) {
            __m256d akp = _mm256_loadu_pd(&ATA(k, p));
            __m256d akq = _mm256_loadu_pd(&ATA(k, q));
            __m256d new_akp = _mm256_fmadd_pd(cos_vec, akp, _mm256_mul_pd(sin_vec, akq));
            __m256d new_akq = _mm256_fnmadd_pd(sin_vec, akp, _mm256_mul_pd(cos_vec, akq));
            _mm256_storeu_pd(&ATA(k, p), new_akp);
            _mm256_storeu_pd(&ATA(k, q), new_akq);
        }

        // 更新 V
        for (size_t k = 0; k < n; k += 4) {
            __m256d vkp = _mm256_loadu_pd(&V(k, p));
            __m256d vkq = _mm256_loadu_pd(&V(k, q));
            __m256d new_vkp = _mm256_fmadd_pd(cos_vec, vkp, _mm256_mul_pd(sin_vec, vkq));
            __m256d new_vkq = _mm256_fnmadd_pd(sin_vec, vkp, _mm256_mul_pd(cos_vec, vkq));
            _mm256_storeu_pd(&V(k, p), new_vkp);
            _mm256_storeu_pd(&V(k, q), new_vkq);
        }
    }

    // Step 3: 提取奇異值
    for (size_t i = 0; i < n; ++i) {
        S(i, i) = std::sqrt(std::max(ATA(i, i), 0.0)); // 確保非負數
    }

    // Step 4: 計算 U = A * V * S^-1
    Matrix Sigma_inv(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (S(i, i) > 1e-10) {
            Sigma_inv(i, i) = 1.0 / S(i, i);
        }
    }
    U = matrix_multiply_simd(A, matrix_multiply_simd(V, Sigma_inv));

    return {U, S, V};
}