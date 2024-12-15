## 1. Introduction
本文件提供了矩陣運算相關功能的 API 說明，包括矩陣乘法、SVD 分解、PCA 降維等核心功能。本函式庫支援 Naive、SIMD 與 Strassen 等不同演算法實作的SVD/Matrix/Multplication Pca，適合進行相關的數值計算。

## 2. API 
Basic Introduce
```python
    Matrix(nrow, ncol, data) #Build up a matrix with data
    #for example
    from Matrix import Matrix

    # Build 3x3 matrix
    A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
```
API-matrix multplication
```python
    matrix_multiply_naive(A, B) #naive multiplication
    simd_matrix_multiply(A, B) #Speed up by SIMD
    Strassen(A, B, 32) #Speed up by Strassen Algo
```

API-SVD decomposition
```python
    svd_jacobi(A, use_strassen=False) #svd decomposition on A with naive multiplication
    svd_jacobi(A, use_strassen=True) #svd decomposition on A with strassen algo
    svd_jacobi_simd(A) #use SIMD decomposition on A to Speed up Jacobi and multplication in SVD decomposition
```

API-Pca
```python
    PCA(A, k, True) #PCA on Matrix A and Get the top k most important information; true means use Strassen to speed up multplu in PCA
```