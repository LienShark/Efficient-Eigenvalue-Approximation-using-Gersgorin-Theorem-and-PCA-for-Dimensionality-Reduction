#include "include/Matrix.hpp"
#include "include/SVD.hpp"
#include "include/PCA.hpp"
#include "include/SIMD_Optimized.hpp"
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>
#include <thread>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <vector>

namespace py=pybind11;

Matrix::Matrix(){
    this->m_nrow = 0;
    this->m_ncol = 0;
    this->m_buffer = nullptr;
}

Matrix::Matrix(size_t nrow, size_t ncol){
    this->m_nrow = nrow;
    this->m_ncol = ncol;
    this->m_buffer = new double[nrow * ncol];
    for(size_t i = 0; i < nrow * ncol; i++){
        this->m_buffer[i] = 0;
    }
}

Matrix::Matrix(size_t row, size_t col, double val){
    this->m_nrow = row;
    this->m_ncol = col;
    this->m_buffer = new double[row * col];
    for(size_t i = 0; i < row * col; i++){
        this->m_buffer[i] = val;
    }
}

Matrix::Matrix(size_t row, size_t col,const std::vector<double> &v){
    this->m_nrow = row;
    this->m_ncol = col;
    this->m_buffer = new double[row * col];
    if(v.size() != row * col){
        throw std::invalid_argument("size of vector does not match matrix size");
    }
    for(size_t i = 0; i < row * col; i++){
        this->m_buffer[i] = v[i];
    }
}
Matrix::Matrix(const Matrix &m){
    this->m_nrow = m.m_nrow;
    this->m_ncol = m.m_ncol;
    this->m_buffer = new double[m.m_nrow * m.m_ncol];
    for(size_t i = 0; i < m.m_nrow * m.m_ncol; i++){
        this->m_buffer[i] = m.m_buffer[i];
    }
} 

size_t Matrix::index(size_t i, size_t j) const{
    return i * m_ncol + j;
}
size_t Matrix::nrow() const{
    return m_nrow;
}
size_t Matrix::ncol() const{
    return m_ncol;
}

double* Matrix::get_buffer() const{
    return m_buffer;
}
    
Matrix::~Matrix() {
    delete[] m_buffer;
}

double Matrix::operator() (size_t row, size_t col) const{
    if (row < 0 || row >= m_nrow || col < 0 || col > m_ncol){
        throw std::out_of_range("index out of range");
    }
    return m_buffer[index(row, col)];
}
double &Matrix::operator() (size_t row, size_t col){
    if (row < 0 || row >= m_nrow || col < 0 || col > m_ncol){
        throw std::out_of_range("index out of range");
    }
    return m_buffer[index(row, col)];
}
bool Matrix::operator==(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        return false;
    }
    for(size_t i = 0; i < this->m_nrow * this->m_ncol; i++){
        if(this->m_buffer[i] != m.m_buffer[i]){
            return false;
        }
    }
    return true;
}

Matrix Matrix::operator+(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < m_nrow * m_ncol; i++){
        result.m_buffer[i] = this->m_buffer[i] + m.m_buffer[i];
    }
    return result;
}
Matrix Matrix::operator-(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < m_nrow * m_ncol; i++){
        result.m_buffer[i] = this->m_buffer[i] - m.m_buffer[i];
    }
    return result;

}

Matrix& Matrix::operator=(const Matrix &m) {
    // 如果維度不匹配，重新分配內存
    if (this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol) {
        delete[] m_buffer; // 釋放原內存
        m_nrow = m.m_nrow;
        m_ncol = m.m_ncol;
        m_buffer = new double[m_nrow * m_ncol]; // 分配新內存
    }

    // 複製數據
    for (size_t i = 0; i < m_nrow * m_ncol; i++) {
        this->m_buffer[i] = m.m_buffer[i];
    }
    return *this;
}


Matrix Matrix::transpose() const {
    Matrix transposed(m_ncol, m_nrow);
    for (size_t i = 0; i < m_nrow; ++i) {
        for (size_t j = 0; j < m_ncol; ++j) {
            transposed(j, i) = (*this)(i, j);
        }
    }
    return transposed;
}

Matrix matrix_multiply_naive(Matrix const &m1, Matrix const &m2){
    if(m1.ncol() != m2.nrow()){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m1.nrow(), m2.ncol());
    for(size_t i = 0; i < m1.nrow(); i++){
        for(size_t j = 0; j < m2.ncol(); j++){
            for(size_t k = 0; k < m1.ncol(); k++){
                result(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
    return result;
}
Matrix matrix_multiply_naive_tile(Matrix const &m1, Matrix const &m2, std::size_t size){
    if(m1.ncol() != m2.nrow()){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m1.nrow(), m2.ncol());
    for(size_t i = 0; i < m1.nrow(); i += size){
        for(size_t j = 0; j < m2.ncol(); j += size){
            for(size_t k = 0; k < m1.ncol(); k += size){
                for(size_t ii = i; ii < std::min(i + size, m1.nrow()); ii++){
                    for(size_t jj = j; jj < std::min(j + size, m2.ncol()); jj++){
                        for(size_t kk = k; kk < std::min(k + size, m1.ncol()); kk++){
                            result(ii, jj) += m1(ii, kk) * m2(kk, jj);
                        }
                    }
                }
            }
        }
    }
    return result;

}

Matrix Matrix::sqrt() const {
    Matrix result(m_nrow, m_ncol);
    for (size_t i = 0; i < m_nrow * m_ncol; ++i) {
        if (m_buffer[i] < 0) {
            throw std::invalid_argument("Negative value encountered in sqrt()");
        }
        result.m_buffer[i] = std::sqrt(m_buffer[i]);
    }
    return result;
}

Matrix Matrix::inverse() const {
    Matrix result(m_nrow, m_ncol);
    for (size_t i = 0; i < m_nrow * m_ncol; ++i) {
        if (m_buffer[i] == 0) {
            throw std::invalid_argument("Cannot invert matrix with zero singular values");
        }
        result.m_buffer[i] = 1.0 / m_buffer[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(m_nrow, m_ncol);
    for (size_t i = 0; i < m_nrow * m_ncol; ++i) {
        result.m_buffer[i] = this->m_buffer[i] * scalar;
    }
    return result;
}

Matrix submatrix(const Matrix &matrix, size_t row_start, size_t col_start, size_t row_end, size_t col_end) {
    // 檢查範圍
    if (row_end > matrix.nrow() || col_end > matrix.ncol()) {
        throw std::out_of_range("Submatrix indices are out of range");
    }
    if (row_start >= row_end || col_start >= col_end) {
        throw std::invalid_argument("Invalid submatrix range");
    }

    // 計算子矩陣的大小
    size_t sub_rows = row_end - row_start;
    size_t sub_cols = col_end - col_start;
    Matrix submatrix(sub_rows, sub_cols);

    // 複製對應的值到子矩陣
    for (size_t i = 0; i < sub_rows; ++i) {
        for (size_t j = 0; j < sub_cols; ++j) {
            submatrix(i, j) = matrix(row_start + i, col_start + j); // 使用 operator() 存取數據
        }
    }

    return submatrix;
}


Matrix strassen_matrix_multiply(const Matrix &m1, const Matrix &m2, size_t block_size) {
    if (m1.ncol() != m2.nrow()) {
        throw std::invalid_argument("Matrix size does not match");
    }

    // 確保矩陣為正方形且大小為 2 的冪次
    size_t n = std::max({m1.nrow(), m1.ncol(), m2.ncol()});
    size_t new_size = 1;
    while (new_size < n) {
        new_size *= 2; // 找到最近的 2 的冪次
    }

    // 將矩陣填充為正方形矩陣
    Matrix A_padded(new_size, new_size);
    Matrix B_padded(new_size, new_size);

    // 填充 A 和 B
    for (size_t i = 0; i < m1.nrow(); ++i) {
        for (size_t j = 0; j < m1.ncol(); ++j) {
            A_padded(i, j) = m1(i, j);
        }
    }
    for (size_t i = 0; i < m2.nrow(); ++i) {
        for (size_t j = 0; j < m2.ncol(); ++j) {
            B_padded(i, j) = m2(i, j);
        }
    }

    // 執行 Strassen 演算法
    Matrix C_padded = strassen_recursive(A_padded, B_padded, block_size);

    // 提取原始大小的結果
    Matrix result(m1.nrow(), m2.ncol());
    for (size_t i = 0; i < result.nrow(); ++i) {
        for (size_t j = 0; j < result.ncol(); ++j) {
            result(i, j) = C_padded(i, j);
        }
    }

    return result;
}

Matrix strassen_recursive(const Matrix &A, const Matrix &B, size_t block_size) {
    size_t n = A.nrow();

    // 當矩陣大小小於 block_size 時，使用基本乘法
    if (n <= block_size) {
        return matrix_multiply_naive_tile(A, B, block_size);
    }

    size_t half = n / 2;

    // 分割矩陣
    Matrix A11 = submatrix(A, 0, 0, half, half);
    Matrix A12 = submatrix(A, 0, half, half, n);
    Matrix A21 = submatrix(A, half, 0, n, half);
    Matrix A22 = submatrix(A, half, half, n, n);

    Matrix B11 = submatrix(B, 0, 0, half, half);
    Matrix B12 = submatrix(B, 0, half, half, n);
    Matrix B21 = submatrix(B, half, 0, n, half);
    Matrix B22 = submatrix(B, half, half, n, n);

    // Strassen 的 7 個子矩陣
    Matrix M1 = strassen_recursive(A11 + A22, B11 + B22, block_size);
    Matrix M2 = strassen_recursive(A21 + A22, B11, block_size);
    Matrix M3 = strassen_recursive(A11, B12 - B22, block_size);
    Matrix M4 = strassen_recursive(A22, B21 - B11, block_size);
    Matrix M5 = strassen_recursive(A11 + A12, B22, block_size);
    Matrix M6 = strassen_recursive(A21 - A11, B11 + B12, block_size);
    Matrix M7 = strassen_recursive(A12 - A22, B21 + B22, block_size);

    // 合併結果
    Matrix C11 = M1 + M4 - M5 + M7;
    Matrix C12 = M3 + M5;
    Matrix C21 = M2 + M4;
    Matrix C22 = M1 - M2 + M3 + M6;

    Matrix result(n, n);
    for (size_t i = 0; i < half; ++i) {
        for (size_t j = 0; j < half; ++j) {
            result(i, j) = C11(i, j);
            result(i, j + half) = C12(i, j);
            result(i + half, j) = C21(i, j);
            result(i + half, j + half) = C22(i, j);
        }
    }

    return result;
}




PYBIND11_MODULE(Matrix, m) {
    py::class_<Matrix>(m, "Matrix")
    .def(py::init<>())
    .def(py::init<size_t, size_t>())
    .def(py::init<size_t, size_t, double>())
    .def(py::init<size_t, size_t, const std::vector<double> &>())
    .def(py::init<const Matrix &>())
    .def("__getitem__", [](Matrix &m, std::vector<std::size_t> idx){ 	 
        return m(idx[0],idx[1]);       
    })
    .def("__setitem__",[](Matrix &m, std::vector<std::size_t> idx, int val){
        m(idx[0],idx[1]) = val;
    })
    .def_property_readonly("nrow", &Matrix::nrow)
    .def_property_readonly("ncol", &Matrix::ncol)
    .def("__str__", [](const Matrix &mat) {
            std::ostringstream oss;
            for (size_t i = 0; i < mat.nrow(); ++i) {
                for (size_t j = 0; j < mat.ncol(); ++j) {
                    oss << mat(i, j) << " ";
                }
                oss << "\n";
            }
            return oss.str();
        })
    .def("__eq__", &Matrix::operator ==);

    m.def("matrix_multiply_naive", &matrix_multiply_naive, "");
    m.def("matrix_multiply_naive_tile", &matrix_multiply_naive_tile, "");
    m.def("svd_jacobi", &svd_jacobi, "Perform SVD using Jacobi method");
    m.def("jacobi_eigen", &jacobi_eigen, "Perform eigenvalue decomposition using Jacobi method");
    m.def("PCA" , &PCA , "Perform PCA");
    m.def("Submatrix" , &submatrix , "Cut submatrix");
    m.def("Strassen" , &strassen_matrix_multiply , "Perform matrix multplication by strassen");
    m.def("simd_matrix_multiply", &matrix_multiply_simd, "Perform matrix multiplication using SIMD optimization");
    m.def("svd_jacobi_simd" , &svd_jacobi_simd , "Perform SVD using SIMD");
}