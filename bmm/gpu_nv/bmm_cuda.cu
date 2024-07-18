#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

const int N = 1024;
const int BLOCK_SIZE = 32;

__global__ void blockedMatrixMultiplyKernel(const int *A, const int *B, int *C, int n) {
    __shared__ int sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    int sum = 0;

    for (int m = 0; m < n / BLOCK_SIZE; ++m) {
        sA[ty][tx] = A[row * n + (m * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * n + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

void initializeMatrix(std::vector<int>& matrix) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = i % 10; // Simple initialization
    }
}

void cudaBlockedMatrixMultiply(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C) {
    int *d_A, *d_B, *d_C;
    int size = N * N * sizeof(int);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);

    blockedMatrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void printPartialMatrix(const std::vector<int>& matrix) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << matrix[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> A(N * N), B(N * N), C(N * N, 0);

    initializeMatrix(A);
    initializeMatrix(B);

    std::cout << "Partial Matrix A:" << std::endl;
    printPartialMatrix(A);
    std::cout << "Partial Matrix B:" << std::endl;
    printPartialMatrix(B);

    auto start = std::chrono::high_resolution_clock::now();

    cudaBlockedMatrixMultiply(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Partial Result Matrix C:" << std::endl;
    printPartialMatrix(C);

    std::cout << "Total computation time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
