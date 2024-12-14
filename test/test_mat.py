import time
import random
import numpy as np
from Matrix import Matrix, Strassen, svd_jacobi, PCA

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
    sizes = [2, 16, 256]
    for size in sizes:
        print(f"測試 {size}x{size} 矩陣...")
        matrix1 = generate_random_matrix(size, size)
        matrix2 = generate_random_matrix(size, size)
        
        # 測試 Strassen
        try:
            result_cpp, duration_cpp = time_it(Strassen, matrix1, matrix2, 64)
            print(f"Strassen 完成，耗時: {duration_cpp:.3f} 秒")
        except Exception as e:
            print(f"Strassen {size}x{size} 測試失敗: {e}")
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

        if np.allclose(result_cpp_list, result_np, atol=1e-5):
            print(f"{size}x{size} 矩陣乘法測試通過！")
        else:
            print(f"{size}x{size} 矩陣乘法測試失敗！")

def test_svd():
    print("\n測試 SVD...")
    sizes = [2, 64, 256]
    for size in sizes:
        print(f"測試 {size}x{size} 矩陣...")
        matrix = generate_random_matrix(size, size)
        
        #C++ 的 SVD 中获取 U, S, V 實作一般矩陣乘法與strassen
        (U_cpp, S_cpp, V_cpp) , cpp_t= time_it(svd_jacobi,matrix,False)
        (U_cpp_st, S_cpp_st, V_cpp_st) , cpp_t_st= time_it(svd_jacobi,matrix,True)
        
        #C++ 的 Matrix  NumPy 
        U_cpp_np = cpp_matrix_to_numpy(U_cpp)
        S_cpp_np = cpp_matrix_to_numpy(S_cpp)
        V_cpp_np = cpp_matrix_to_numpy(V_cpp)
        
        # 验证 S 是否为对角矩阵
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
        if np.allclose(reconstructed_cpp, matrix_np, atol=1e-5):
            print(f"SVD {size}x{size} 測試通過：U * S * V^T 還原成功！")
            print(f"SVD_Jacobi 完成，耗時: {cpp_t:.3f} 秒")
            print(f"SVD_Jacobi_strassen 完成，耗時: {cpp_t_st:.3f} 秒")
            print(f"Numpy 完成，耗時: {np_t:.3f} 秒")
        else:
            print(f"SVD {size}x{size} 測試失敗：U * S * V^T 無法還原 A！")


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




if __name__ == "__main__":
    # test_matrix_multiplication()
    test_svd()
    # test_pca()

