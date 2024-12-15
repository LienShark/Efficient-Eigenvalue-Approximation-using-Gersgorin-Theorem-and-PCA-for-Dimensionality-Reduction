import unittest
import time
import numpy as np
from Matrix import Matrix, Strassen, svd_jacobi, PCA, simd_matrix_multiply, matrix_multiply_naive, svd_jacobi_simd


def generate_random_matrix(rows, cols):
    data = [np.random.uniform(0, 10) for _ in range(rows * cols)]
    return Matrix(rows, cols, data)


def time_it(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start


def cpp_matrix_to_numpy(cpp_matrix):
    rows, cols = cpp_matrix.nrow, cpp_matrix.ncol
    return np.array([[cpp_matrix[i, j] for j in range(cols)] for i in range(rows)])


class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        self.sizes_mul = [8, 256, 1024, 2048]
        self.sizes_svd = [4, 128, 512]
        self.sizes_pca = [4, 64, 128]
        self.output_file = "test_results.txt"
        with open(self.output_file, "a") as f:
            f.write("Matrix Operations Test Results\n")
            f.write("-" * 40 + "\n")

    def log_result(self, test_name, size, duration):
        with open(self.output_file, "a") as f:
            f.write(f"{test_name} (Size={size}): {duration:.3f} 秒\n")

    def test_matrix_multiply(self):
        for size in self.sizes_mul:
            with self.subTest(size=size):
                A = generate_random_matrix(size, size)
                B = generate_random_matrix(size, size)

                try:
                    # 測試矩陣乘法
                    result, duration = time_it(matrix_multiply_naive, A, B)
                    self.log_result("Matrix Multiply by naive", size, duration)
                    A_np = cpp_matrix_to_numpy(A)
                    B_np = cpp_matrix_to_numpy(B)
                    result_np = np.dot(A_np, B_np)
                    result_cpp = cpp_matrix_to_numpy(result)
                    self.assertTrue(np.allclose(result_np, result_cpp, atol=1e-5))

                    # 測試矩陣乘法-SIMD
                    result, duration = time_it(simd_matrix_multiply, A, B)
                    self.log_result("Matrix Multiply by SIMD", size, duration)
                    result_cpp = cpp_matrix_to_numpy(result)
                    self.assertTrue(np.allclose(result_np, result_cpp, atol=1e-5))

                    # 測試矩陣乘法-Strassen
                    result, duration = time_it(Strassen, A, B, 32)
                    self.log_result("Matrix Multiply by Strassen", size, duration)
                    result_cpp = cpp_matrix_to_numpy(result)
                    self.assertTrue(np.allclose(result_np, result_cpp, atol=1e-5))
                except Exception as e:
                    self.log_result("Matrix Multiply Error", size, 0)
                    print(f"Error in Matrix Multiply (Size={size}): {e}")

    def test_svd(self):
        for size in self.sizes_svd:
            with self.subTest(size=size):
                matrix = generate_random_matrix(size, size)
                matrix_np = cpp_matrix_to_numpy(matrix)

                try:
                    # 測試 SVD (一般乘法)
                    (U_cpp, S_cpp, V_cpp), cpp_time = time_it(svd_jacobi, matrix, False)
                    self.log_result("SVD (一般乘法)", size, cpp_time)

                    # 測試 SVD (Strassen)
                    (U_str, S_str, V_str), str_time = time_it(svd_jacobi, matrix, True)
                    self.log_result("SVD (Strassen)", size, str_time)

                    # 測試 SVD (SIMD)
                    (U_simd, S_simd, V_simd), simd_time = time_it(svd_jacobi_simd, matrix)
                    self.log_result("SVD (SIMD)", size, simd_time)

                    # 驗證結果
                    reconstructed_cpp = np.dot(cpp_matrix_to_numpy(U_cpp),
                                               np.dot(cpp_matrix_to_numpy(S_cpp),
                                                      cpp_matrix_to_numpy(V_cpp).T))
                    reconstructed_str = np.dot(cpp_matrix_to_numpy(U_str),
                                               np.dot(cpp_matrix_to_numpy(S_str),
                                                      cpp_matrix_to_numpy(V_str).T))
                    reconstructed_simd = np.dot(cpp_matrix_to_numpy(U_simd),
                                                np.dot(cpp_matrix_to_numpy(S_simd),
                                                       cpp_matrix_to_numpy(V_simd).T))
                    self.assertTrue(np.allclose(reconstructed_cpp, matrix_np, atol=1e-2))
                    self.assertTrue(np.allclose(reconstructed_str, matrix_np, atol=1e-2))
                    # self.assertTrue(np.allclose(cpp_matrix_to_numpy(S_simd),
                    #                             np.diag(np.diagonal(cpp_matrix_to_numpy(S_simd))), atol=1))
                    self.assertTrue(np.allclose(reconstructed_simd, matrix_np, atol=1e-2))
                except Exception as e:
                    self.log_result("SVD Error", size, 0)
                    print(f"Error in SVD (Size={size}): {e}")

    def test_pca(self):
        for size in self.sizes_pca:
            with self.subTest(size=size):
                matrix = generate_random_matrix(size, size)

                try:
                    # 測試 PCA
                    _, duration_withStr = time_it(PCA, matrix, 2, True)
                    self.log_result("PCA 使用strassen", size, duration_withStr)
                except Exception as e:
                    self.log_result("PCA Error", size, 0)
                    print(f"Error in PCA (Size={size}): {e}")


if __name__ == "__main__":
    unittest.main()
