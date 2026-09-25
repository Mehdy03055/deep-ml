#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

std::vector<float> matrix_dot_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    // Check dimension compatibility
    if (cols != vec.size()) {
        return {-1};
    }
    
    // Flatten matrix to 1D array
    std::vector<float> flat_matrix(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat_matrix[i * cols + j] = matrix[i][j];
        }
    }
    
    float *d_matrix, *d_vector, *d_result;
    std::vector<float> h_result(rows);
    
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_vector, cols * sizeof(float));
    cudaMalloc(&d_result, rows * sizeof(float));
    
    cudaMemcpy(d_matrix, flat_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vec.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    matrix_vector_dot_kernel<<<blocks, threads>>>(d_matrix, d_vector, d_result, rows, cols);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result.data(), d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    
    return h_result;
}