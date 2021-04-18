// CUDA operations fro matrix multiply
// References: 	https://stackoverflow.com/questions/35799478/how-to-implement-a-nxm-cuda-matrix-multiplication
// 				https://stackoverflow.com/questions/18997773/non-square-matrix-multiplication-in-cuda
#include <cuda_runtime_api.h>
#include <cuda.h>

#define SHARED_MEM_TILE_WIDTH 16

template <typename T>
__global__ void CUDA_MAT_MULT_NORMAL(T *d_A, T *d_B, T *d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols) {

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

template <typename T>
__global__ void CUDA_MAT_MULT_TILED(T* d_A, T* d_B, T* d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols, int TILE_WIDTH) {

	T c_val = 0;

	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	for (int i = 0; i < (TILE_WIDTH + A_cols - 1)/TILE_WIDTH; i++)
	{
		for (int k = 0; k < TILE_WIDTH; k++)
		{
			if ((i * TILE_WIDTH + k < A_cols && row < A_rows) && (i * TILE_WIDTH + k < B_rows && col < B_cols))
			{
				c_val += d_A[row * A_cols + i * TILE_WIDTH + k] * d_B[(i * TILE_WIDTH + k) * B_cols + col];
			}
		}
	}

	if (row < C_rows && col < C_cols)
	{
		d_C[((blockIdx.y * blockDim.y + threadIdx.y) * C_cols) + (blockIdx.x * blockDim.x) + threadIdx.x] = c_val;
	}
}

template <typename T>
__global__ void CUDA_MAT_MULT_SHARED_NORMAL(T* d_A, T* d_B, T* d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols)
{
	T c_val = 0;

	int row = blockIdx.y*SHARED_MEM_TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x*SHARED_MEM_TILE_WIDTH + threadIdx.x;

	__shared__ T s_A[SHARED_MEM_TILE_WIDTH][SHARED_MEM_TILE_WIDTH];
	__shared__ T s_B[SHARED_MEM_TILE_WIDTH][SHARED_MEM_TILE_WIDTH];

	for (int k = 0; k < (SHARED_MEM_TILE_WIDTH + A_cols - 1)/SHARED_MEM_TILE_WIDTH; k++) {
		if (k*SHARED_MEM_TILE_WIDTH + threadIdx.x < A_cols && row < A_rows)
		{
			s_A[threadIdx.y][threadIdx.x] = d_A[row*A_cols + k*SHARED_MEM_TILE_WIDTH + threadIdx.x];
		}
		else
		{
			s_A[threadIdx.y][threadIdx.x] = 0.0;
		}

		if (k*SHARED_MEM_TILE_WIDTH + threadIdx.y < B_rows && col < B_cols)
		{
			s_B[threadIdx.y][threadIdx.x] = d_B[(k*SHARED_MEM_TILE_WIDTH + threadIdx.y)*B_cols + col];
		}
		else
		{
			s_B[threadIdx.y][threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < SHARED_MEM_TILE_WIDTH; ++n)
			c_val += s_A[threadIdx.y][n] * s_B[n][threadIdx.x];

		__syncthreads();
	}

	if (row < C_rows && col < C_cols)
	{
		d_C[((blockIdx.y * blockDim.y + threadIdx.y)*C_cols) + (blockIdx.x * blockDim.x)+ threadIdx.x] = c_val;
	}
}