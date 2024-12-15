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
    return {U, S, V};
}