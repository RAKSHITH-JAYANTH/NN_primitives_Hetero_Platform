#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

const int N = 1024;
const int BLOCK_SIZE = 32;

void initializeMatrix(std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = i + j;
        }
    }
}

void blockedMatrixMultiply(const std::vector<std::vector<int>>& A,
                           const std::vector<std::vector<int>>& B,
                           std::vector<std::vector<int>>& C) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Multiply block
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj++) {
                        for (int kk = k; kk < std::min(k + BLOCK_SIZE, N); kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
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

    blockedMatrixMultiply(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Partial Result Matrix C:" << std::endl;
    printPartialMatrix(C);

    std::cout << "Total computation time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
