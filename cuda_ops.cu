#include <cuda_runtime_api.h>
#include <cuda.h>

// Global CUDA
template <typename T>
__global__ void CUDA_MAT_MULT(T* d_A, T* d_B, T* d_C, int rows_A, int cols_A, int rows_B, int cols_B) {

    T c_val = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < (cols_A); i++) {
            if ((i < cols_A && row < rows_A) && (i < rows_B && col < cols_B))
                c_val += d_A[row*cols_A + i] * d_B[i*cols_B + col];

    }

    if (row < rows_A && col < cols_B){
    	d_C[((blockIdx.y * blockDim.y + threadIdx.y)*cols_B)+(blockIdx.x*blockDim.x)+threadIdx.x] = c_val;
    }
}
