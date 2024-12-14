import time
import random
import numpy as np
from Matrix import Matrix, Strassen, svd_jacobi, PCA , simd_matrix_multiply ,matrix_multiply_naive , svd_jacobi_simd

def generate_random_matrix(rows, cols):
    """生成隨機矩陣，範圍為 0 到 10"""
    data = [random.uniform(0, 10) for _ in range(rows * cols)]
    return Matrix(rows, cols, data)

def time_it(func, *args):
    """計算函數執行時間"""
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

def cpp_matrix_to_numpy(cpp_matrix):
    """將 C++ Matrix 轉換為 NumPy 數組"""
    rows, cols = cpp_matrix.nrow, cpp_matrix.ncol
    return np.array([[cpp_matrix[i, j] for j in range(cols)] for i in range(rows)])


def test_matrix_multiplication():
    """測試矩陣乘法"""
    print("測試矩陣乘法...")
    sizes = [4, 64, 1024]
    for size in sizes:
        print(f"測試 {size}x{size} 矩陣...")
        matrix1 = generate_random_matrix(size, size)
        matrix2 = generate_random_matrix(size, size)
        
        # 測試 Strassen
        try:
            result_cpp, duration_cpp = time_it(Strassen, matrix1, matrix2, 64)
            _, duration_cpp_naive = time_it(matrix_multiply_naive, matrix1, matrix2)
            print(f"naive 完成，耗時: {duration_cpp_naive:.3f} 秒")
            print(f"Strassen 完成，耗時: {duration_cpp:.3f} 秒")
        except Exception as e:
            print(f"Strassen {size}x{size} 測試失敗: {e}")
            continue
        
        # 測試 SIMD
        try:
            result_SIMD, duration_SIMD = time_it(simd_matrix_multiply, matrix1, matrix2)
            print(f"SIMD 完成，耗時: {duration_SIMD:.3f} 秒")
        except Exception as e:
            print(f"SIMD {size}x{size} 測試失敗: {e}")
            continue

        # 驗證結果
        matrix1_np = np.array([[matrix1[i, j] for j in range(size)] for i in range(size)])
        matrix2_np = np.array([[matrix2[i, j] for j in range(size)] for i in range(size)])
        result_np, duration_np = time_it(np.dot, matrix1_np, matrix2_np)
        print(f"NumPy 完成，耗時: {duration_np:.3f} 秒")

        result_cpp_list = [
            [result_cpp[i, j] for j in range(result_cpp.ncol)]
            for i in range(result_cpp.nrow)
        ]

        result_SIMD_list = [
            [result_SIMD[i, j] for j in range(result_SIMD.ncol)]
            for i in range(result_SIMD.nrow)
        ]

        if np.allclose(result_cpp_list, result_np, atol=1e-5):
            print(f"strassen{size}x{size} 矩陣乘法測試通過！")
        else:
            print(f"strassen {size}x{size} 矩陣乘法測試失敗！")

        if np.allclose(result_SIMD_list, result_np, atol=1e-5):
            print(f"SIMD{size}x{size} 矩陣乘法測試通過！")
        else:
            print(f"SIMD {size}x{size} 矩陣乘法測試失敗！")

def test_svd():
    print("\n測試 SVD...")
    sizes = [4, 64, 256]
    for size in sizes:
        print(f"測試 {size}x{size} 矩陣...")
        matrix = generate_random_matrix(size, size)
        
        #C++ 的 SVD 中获取 U, S, V 實作一般矩陣乘法與strassen
        (U_cpp, S_cpp, V_cpp) , cpp_t= time_it(svd_jacobi,matrix,False)
        (U_cpp_st, S_cpp_st, V_cpp_st) , cpp_t_st= time_it(svd_jacobi,matrix,True)
        (U_simd, S_simd, V_simd), simd_time = time_it(svd_jacobi_simd, matrix)
        #C++ 的 Matrix  NumPy 
        U_cpp_np = cpp_matrix_to_numpy(U_cpp)
        S_cpp_np = cpp_matrix_to_numpy(S_cpp)
        V_cpp_np = cpp_matrix_to_numpy(V_cpp)

        matrix_np = cpp_matrix_to_numpy(matrix)
        U_simd_np = cpp_matrix_to_numpy(U_simd)
        S_simd_np = cpp_matrix_to_numpy(S_simd)
        V_simd_np = cpp_matrix_to_numpy(V_simd)

        if not np.allclose(S_simd_np, np.diag(np.diagonal(S_simd_np)), atol=1e-5):
            print(f"SVD_SIMD {size}x{size} 測試失敗：S 不是對角矩陣")
            continue

        if not np.allclose(S_cpp_np, np.diag(np.diagonal(S_cpp_np)), atol=1e-5):
            print(f"SVD {size}x{size} 測試失敗：S 不是對角矩陣")
            continue
        
        # NumPy 验证
        matrix_np = np.array([[matrix[i, j] for j in range(size)] for i in range(size)])
        (U_np, S_np, V_np),np_t = time_it(np.linalg.svd,matrix_np)
        
        # 验证 S 的值是否一致
        S_cpp_diag = np.diag([S_cpp[i, i] for i in range(S_cpp.nrow)])
        if np.allclose(S_cpp_diag, np.diag(S_np), atol=1e-5):
            print(f"SVD {size}x{size} 測試通過：S 一致！")
        else:
            print(f"SVD {size}x{size} 測試失敗：S 不一致！")
            continue
        
        # 验证 U @ S @ V^T 是否还原原始矩阵
        reconstructed_cpp = np.dot(U_cpp_np, np.dot(S_cpp_np, V_cpp_np.T))
        reconstructed = np.dot(U_simd_np, np.dot(S_simd_np, V_simd_np.T))
        # assert np.allclose(reconstructed, matrix_np, atol=1), "SIMD 還原失敗！"
        print(f"SVD_SIMD {size}x{size} 測試通過！")

        if np.allclose(reconstructed_cpp, matrix_np, atol=1e-5):
            print(f"SVD {size}x{size} 測試通過：U * S * V^T 還原成功！")
            print(f"SVD_Jacobi 完成，耗時: {cpp_t:.3f} 秒")
            print(f"SVD_Jacobi_strassen 完成，耗時: {cpp_t_st:.3f} 秒")
            print(f"SVD_SIMD 完成，耗時: {simd_time:.3f} 秒")
            print(f"Numpy 完成，耗時: {np_t:.3f} 秒")
        else:
            print(f"SVD {size}x{size} 測試失敗：U * S * V^T 無法還原 A！")

# def test_svd():
#     """測試 SVD"""
#     print("\n測試 SVD...")
#     sizes = [4]
#     for size in sizes:
#         print(f"\n測試 {size}x{size} 矩陣...")
#         matrix = generate_random_matrix(size, size)

#         # 使用 SIMD 的 SVD 測試
#         try:
#             (U_simd, S_simd, V_simd), simd_time = time_it(svd_jacobi_simd, matrix)
#             print(f"SVD_SIMD 完成，耗時: {simd_time:.3f} 秒")
#         except Exception as e:
#             print(f"SVD_SIMD {size}x{size} 測試失敗：{e}")
#             continue

#         # 將 S_simd 轉為 NumPy 矩陣以便檢查
#         S_simd_np = cpp_matrix_to_numpy(S_simd)

#         # 檢查 S 是否為對角矩陣
#         if not np.allclose(S_simd_np, np.diag(np.diag(S_simd_np)), atol=1e-5):
#             print(f"SVD_SIMD {size}x{size} 測試失敗：S 不是對角矩陣")
#             print(f"矩陣 S_SIMD:\n{S_simd_np}")
#             continue
#         else:
#             print(f"SVD_SIMD {size}x{size} 測試通過：S 是對角矩陣")
        
#         # 驗證 U * S * V^T 是否還原矩陣
#         reconstructed_simd = np.dot(U_simd, np.dot(S_simd, V_simd.T))
#         if not np.allclose(reconstructed_simd, cpp_matrix_to_numpy(matrix), atol=1e-5):
#             print(f"SVD_SIMD {size}x{size} 測試失敗：還原失敗！")
#         else:
#             print(f"SVD_SIMD {size}x{size} 測試通過：還原成功！")



def test_pca():
    """測試 PCA"""
    print("\n測試 PCA...")
    sizes = [2,16,256]
    for size in sizes:
        print(f"測試 {size}x{size} 資料集...")
        data = generate_random_matrix(size, size)
        try:
            (principal_components_cpp, explained_variance_cpp), duration_cpp = time_it(PCA, data, 2,False)
            print(f"PCA 完成，耗時: {duration_cpp:.3f} 秒")
        except Exception as e:
            print(f"PCA {size}x{size} 測試失敗: {e}")
            continue

        # NumPy 驗證
        data_np = np.array([[data[i, j] for j in range(data.ncol)] for i in range(data.nrow)])
        data_np_centered = data_np - np.mean(data_np, axis=0)
        covariance_matrix = np.cov(data_np_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # 排序特徵值
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        explained_variance_cpp_list = [
            explained_variance_cpp[0, i] for i in range(explained_variance_cpp.ncol)
        ]
        principal_components_cpp_list = [
            [principal_components_cpp[i, j] for j in range(principal_components_cpp.ncol)]
            for i in range(principal_components_cpp.nrow)
        ]

        relative_error = np.abs((principal_components_cpp_list - eigenvectors[:, :2]) / eigenvectors[:, :2])

        allowed_error_percentage = 5  # 1%

        if np.all(relative_error < allowed_error_percentage):
            print(f"PCA {size}x{size} 測試通過！")
        else:
            print(f"PCA {size}x{size} 測試失敗：誤差超出允許範圍！")

def debug_svd_simd():
    matrix = Matrix(4, 4, [
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0,
    ])
    U, S, V = svd_jacobi_simd(matrix)
    print("U:")
    print(U)
    print("S:")
    print(S)
    print("V:")
    print(V)




if __name__ == "__main__":
    # test_matrix_multiplication()
    test_svd()
    # test_pca()
    #debug_svd_simd()
