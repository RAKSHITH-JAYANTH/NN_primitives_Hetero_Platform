#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <cmath>

const int N = 1000; // Larger size for better demonstration
const int BLOCK_SIZE = 50;
const int NUM_THREADS = 4;

struct ThreadData {
    const std::vector<std::vector<int>>* A;
    const std::vector<std::vector<int>>* B;
    std::vector<std::vector<int>>* C;
    int startRow;
    int startCol;
    int endRow;
    int endCol;
};

void initializeMatrix(std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = i + j; // Simple initialization
        }
    }
}

void* multiplyBlock(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    const std::vector<std::vector<int>>& A = *(data->A);
    const std::vector<std::vector<int>>& B = *(data->B);
    std::vector<std::vector<int>>& C = *(data->C);

    for (int i = data->startRow; i < data->endRow; i += BLOCK_SIZE) {
        for (int j = data->startCol; j < data->endCol; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, data->endRow); ++ii) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, data->endCol); ++jj) {
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, N); ++kk) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }

    return NULL;
}

void parallelBlockMatrixMultiply(const std::vector<std::vector<int>>& A,
                                 const std::vector<std::vector<int>>& B,
                                 std::vector<std::vector<int>>& C) {
    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];
    int blockRowSize = N / std::sqrt(NUM_THREADS);
    int blockColSize = N / std::sqrt(NUM_THREADS);

    int threadIndex = 0;
    for (int i = 0; i < N; i += blockRowSize) {
        for (int j = 0; j < N; j += blockColSize) {
            if (threadIndex < NUM_THREADS) {
                threadData[threadIndex].A = &A;
                threadData[threadIndex].B = &B;
                threadData[threadIndex].C = &C;
                threadData[threadIndex].startRow = i;
                threadData[threadIndex].startCol = j;
                threadData[threadIndex].endRow = std::min(i + blockRowSize, N);
                threadData[threadIndex].endCol = std::min(j + blockColSize, N);

                pthread_create(&threads[threadIndex], NULL, multiplyBlock, (void*)&threadData[threadIndex]);
                threadIndex++;
            }
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void printPartialMatrix(const std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < std::min(5, N); i++) {
        for (int j = 0; j < std::min(5, N); j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::vector<std::vector<int>> A(N, std::vector<int>(N));
    std::vector<std::vector<int>> B(N, std::vector<int>(N));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    initializeMatrix(A);
    initializeMatrix(B);

    std::cout << "Partial Matrix A:" << std::endl;
    printPartialMatrix(A);
    std::cout << "Partial Matrix B:" << std::endl;
    printPartialMatrix(B);

    auto start = std::chrono::high_resolution_clock::now();

    parallelBlockMatrixMultiply(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Partial Result Matrix C:" << std::endl;
    printPartialMatrix(C);

    std::cout << "Total computation time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
