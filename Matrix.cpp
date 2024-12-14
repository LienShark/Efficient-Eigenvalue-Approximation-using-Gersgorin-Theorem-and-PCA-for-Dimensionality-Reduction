#include "Matrix.hpp"
#include "SVD.hpp"
#include "PCA.hpp"
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

Matrix matrix_multiply_naive_cache_optimized_tile_thread(const Matrix &m1, const Matrix &m2, size_t block_size) {
    if (m1.ncol() != m2.nrow()) {
        throw std::invalid_argument("matrix size does not match");
    }

    Matrix m2_transposed = m2.transpose();
    Matrix result(m1.nrow(), m2.ncol());

    size_t num_threads = std::thread::hardware_concurrency();
    
    std::vector<std::thread> threads(min(num_threads, m1.nrow()));

    for (size_t t = 0; t < num_threads; t++) {
        threads[t] = std::thread([&](size_t thread_id) {
            size_t start_row = (thread_id * m1.nrow()) / num_threads;
            size_t end_row = ((thread_id + 1) * m1.nrow()) / num_threads;

            for (size_t i = start_row; i < end_row; i += block_size) {
                for (size_t j = 0; j < m2_transposed.nrow(); j += block_size) {
                    for (size_t k = 0; k < m1.ncol(); k += block_size) {
                        for (size_t ii = i; ii < std::min(i + block_size, m1.nrow()); ii++) {
                            for (size_t jj = j; jj < std::min(j + block_size, m2_transposed.nrow()); jj++) {
                                for (size_t kk = k; kk < std::min(k + block_size, m1.ncol()); kk++) {
                                    result(ii, jj) += m1(ii, kk) * m2_transposed(jj, kk);
                                }
                            }
                        }
                    }
                }
            }
        }, t);
    }

    for (size_t t = 0; t < num_threads; t++) {
        threads[t].join();
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
    .def("__eq__", &Matrix::operator ==);

    m.def("matrix_multiply_naive", &matrix_multiply_naive, "");
    m.def("matrix_multiply_naive_tile", &matrix_multiply_naive_tile, "");
    m.def("matrix_multiply_naive_cache_optimized_tile_thread", &matrix_multiply_naive_cache_optimized_tile_thread, "");
    m.def("svd_jacobi", &svd_jacobi, "Perform SVD using Jacobi method");
    m.def("jacobi_eigen", &jacobi_eigen, "Perform eigenvalue decomposition using Jacobi method");
    m.def("PCA" , &PCA , "Perform PCA");
}