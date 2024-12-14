from Matrix import Matrix, svd_jacobi, PCA
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
matrix = Matrix(3, 2, [2.5, 2.4, 0.5, 0.7, 2.2, 2.9])

U, S, V = svd_jacobi(matrix)

# 輸出結果
print("U:")
for i in range(U.nrow):
    print([U[i, j] for j in range(U.ncol)])

print("\nS:")
for i in range(S.nrow):
    print([S[i, j] for j in range(S.ncol)])

print("\nV:")
for i in range(V.nrow):
    print([V[i, j] for j in range(V.ncol)])


# 創建測試資料矩陣 (3x2)
data = Matrix(3, 2, [2.5, 2.4, 0.5, 0.7, 2.2, 2.9])

# 執行 PCA
num_components = 2
principal_components, explained_variance = PCA(data, num_components)

# 輸出主成分和解釋的變異數
print("Principal Components:")
for i in range(principal_components.nrow):
    print([principal_components[i, j] for j in range(principal_components.ncol)])

print("\nExplained Variance:")
for i in range(explained_variance.nrow):
    print([explained_variance[i, j] for j in range(explained_variance.ncol)])
