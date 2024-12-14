from Matrix import Matrix, svd_jacobi, PCA ,Submatrix
# 建立測試矩陣
# matrix = Matrix(3, 3)
# matrix[0, 0] = 3
# matrix[0, 1] = 0
# matrix[0, 2] = 0
# matrix[1, 0] = 0
# matrix[1, 1] = -3
# matrix[1, 2] = 1
# matrix[2, 0] = 0
# matrix[2, 1] = 1
# matrix[2, 2] = -3

# # 執行 Jacobi SVD
# U, S, V = svd_jacobi(matrix)

# # 打印結果
# print("U:")
# for i in range(U.nrow):
#     for j in range(U.ncol):
#         print(f"{U[i, j]:.2f}", end=" ")
#     print()

# print("\nS:")
# for i in range(S.nrow):
#     for j in range(S.ncol):
#         print(f"{S[i, j]:.2f}", end=" ")
#     print()

# print("\nV:")
# for i in range(V.nrow):
#     for j in range(V.ncol):
#         print(f"{V[i, j]:.2f}", end=" ")
#     print()

# 測試矩陣
# matrix = Matrix(3, 2, [3, 2, 2, 3, 2, -2])
# matrix = Matrix(3, 2, [2.5, 2.4, 0.5, 0.7, 2.2, 2.9])

# U, S, V = svd_jacobi(matrix)

# # 輸出結果
# print("U:")
# for i in range(U.nrow):
#     print([U[i, j] for j in range(U.ncol)])

# print("\nS:")
# for i in range(S.nrow):
#     print([S[i, j] for j in range(S.ncol)])

# print("\nV:")
# for i in range(V.nrow):
#     print([V[i, j] for j in range(V.ncol)])


# # 創建測試資料矩陣 (3x2)
# data = Matrix(3, 2, [2.5, 2.4, 0.5, 0.7, 2.2, 2.9])

# # 執行 PCA
# num_components = 2
# principal_components, explained_variance = PCA(data, num_components)

# # 輸出主成分和解釋的變異數
# print("Principal Components:")
# for i in range(principal_components.nrow):
#     print([principal_components[i, j] for j in range(principal_components.ncol)])

# print("\nExplained Variance:")
# for i in range(explained_variance.nrow):
#     print([explained_variance[i, j] for j in range(explained_variance.ncol)])


# mat = Matrix(4,4 , [1,2,3,4,
#                     5,6,7,8,
#                     9,10,11,12,
#                     13,14,15,16])

# submat = Submatrix(mat,1,1,3,3)
# print("\n subnatrix:")
# for i in range(submat.nrow):
#     print([submat[i, j] for j in range(submat.ncol)])
import numpy as np
from Matrix import Matrix, matrix_multiply_naive_tile,Strassen

def test_matrix_multiply_cache_optimized_tile():
    # 定義測試矩陣
    data1 = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    ]
    data2 = [
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    ]

    # 創建 Matrix 對象
    matrix1 = Matrix(3, 3, data1)  # 3x3 矩陣
    matrix2 = Matrix(3, 3, data2)  # 3x3 矩陣

    # 設定 block size
    block_size = 2

    # 調用 C++ 實現的矩陣乘法
    # result_cpp = matrix_multiply_naive_tile(matrix1, matrix2, block_size)
    result_cpp = Strassen(matrix1, matrix2, block_size)

    # 將結果轉換為 Python 列表
    result_cpp_list = [
        [result_cpp[i, j] for j in range(result_cpp.ncol)]
        for i in range(result_cpp.nrow)
    ]

    # 用 NumPy 實現矩陣乘法
    matrix1_np = np.array(data1).reshape(3, 3)  # 將數據轉換為 NumPy 矩陣
    matrix2_np = np.array(data2).reshape(3, 3)
    result_np = np.dot(matrix1_np, matrix2_np)  # NumPy 的矩陣乘法

    # 輸出結果
    print("C++ Matrix Multiplication Result:")
    for row in result_cpp_list:
        print(row)

    print("\nNumPy Matrix Multiplication Result:")
    print(result_np)

    # 檢查結果是否一致
    assert np.allclose(result_cpp_list, result_np), "Results do not match!"
    print("\nTest passed! The results match.")

# 調用測試函數
test_matrix_multiply_cache_optimized_tile()
