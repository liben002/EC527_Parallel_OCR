// CUDA operations fro matrix multiply
// References: 	https://stackoverflow.com/questions/35799478/how-to-implement-a-nxm-cuda-matrix-multiplication
// 				https://stackoverflow.com/questions/18997773/non-square-matrix-multiplication-in-cuda
#include <cuda_runtime_api.h>
#include <cuda.h>

#define SHARED_TILE_WIDTH 16

template <typename T>
__global__ void CUDA_MAT_MULT_NORMAL(T *d_A, T *d_B, T *d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols) {


	// row used for d_A matric, col used for d_B matrix
	T c_val = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < A_cols; i++) {
		if ((i < A_cols && row < A_rows) && (i < B_rows && col < B_cols))
		{
			__syncthreads();
			c_val += d_A[row * A_cols + i] * d_B[i * B_cols + col];
			__syncthreads();
		}
	}

	if (row < C_rows && col < C_cols)
	{
		d_C[((blockIdx.y * blockDim.y + threadIdx.y) * C_cols) + (blockIdx.x * blockDim.x) + threadIdx.x] = c_val;
	}
}

// template <typename T>
// __global__ void CUDA_MAT_MULT_TILED(T* d_A, T* d_B, T* d_C, int rows_A, int cols_A, int rows_B, int cols_B, int rows_C, int cols_C, int TILE_WIDTH) {

// 	T c_val = 0;

// 	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
// 	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

// 	for (int i = 0; i < (TILE_WIDTH + cols_A - 1)/TILE_WIDTH; i++)
// 	{
// 		for (int k = 0; k < TILE_WIDTH; k++)
// 		{
// 			if ((i * TILE_WIDTH + k < cols_A && row < rows_A) && (i * TILE_WIDTH + k < rows_B && col < cols_B))
// 			{
// 				c_val += d_A[row * cols_A + i * TILE_WIDTH + k] * d_B[(i * TILE_WIDTH + k) * cols_B + col];
// 			}
// 		}
// 	}

// 	if (row < rows_C && col < cols_C)
// 	{
// 		d_C[((blockIdx.y * blockDim.y + threadIdx.y) * cols_C) + (blockIdx.x * blockDim.x) + threadIdx.x] = c_val;
// 	}
// }

// template <typename T>
// __global__ void CUDA_MAT_MULT_SHARED_TILED(T *d_A, T *d_B, T *d_C, int rows_A, int cols_A, int rows_B, int cols_B, int rows_C, int cols_C)
// {
// 	T c_val = 0;

// 	int row = blockIdx.y*SHARED_TILE_WIDTH + threadIdx.y;
// 	int col = blockIdx.x*SHARED_TILE_WIDTH + threadIdx.x;

// 	__shared__ T s_A[SHARED_TILE_WIDTH][SHARED_TILE_WIDTH];
// 	__shared__ T s_B[SHARED_TILE_WIDTH][SHARED_TILE_WIDTH];

// 	for (int k = 0; k < (SHARED_TILE_WIDTH + cols_A - 1)/SHARED_TILE_WIDTH; k++) {
// 		if (k*SHARED_TILE_WIDTH + threadIdx.x < cols_A && row < rows_A)
// 		{
// 			s_A[threadIdx.y][threadIdx.x] = d_A[row*cols_A + k*SHARED_TILE_WIDTH + threadIdx.x];
// 		}
// 		else
// 		{
// 			s_A[threadIdx.y][threadIdx.x] = 0.0;
// 		}

// 		if (k*SHARED_TILE_WIDTH + threadIdx.y < rows_B && col < cols_B)
// 		{
// 			s_B[threadIdx.y][threadIdx.x] = d_B[(k*SHARED_TILE_WIDTH + threadIdx.y)*cols_B + col];
// 		}
// 		else
// 		{
// 			s_B[threadIdx.y][threadIdx.x] = 0.0;
// 		}

// 		__syncthreads();

// 		for (int n = 0; n < SHARED_TILE_WIDTH; ++n)
// 			c_val += s_A[threadIdx.y][n] * s_B[n][threadIdx.x];

// 		__syncthreads();
// 	}

// 	if (row < rows_C && col < cols_C)
// 	{
// 		d_C[((blockIdx.y * blockDim.y + threadIdx.y)*cols_C) + (blockIdx.x * blockDim.x)+ threadIdx.x] = c_val;
// 	}
// }