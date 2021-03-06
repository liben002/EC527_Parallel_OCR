// CUDA operations fro matrix multiply
// References: 	https://stackoverflow.com/questions/35799478/how-to-implement-a-nxm-cuda-matrix-multiplication
// 				https://stackoverflow.com/questions/18997773/non-square-matrix-multiplication-in-cuda
#include <cuda_runtime_api.h>
#include <cuda.h>

#define SHARED_TILE_WIDTH 16

template <typename T>
__global__ void CUDA_MAT_MULT_NORMAL(T *d_A, T *d_B, T *d_C, int rows_A, int cols_A, int rows_B, int cols_B, int rows_C, int cols_C)
{

	int row = blockIdx.x * blockDim.x + threadIdx.x; // for d_A matrix
	int col = blockIdx.y * blockDim.y + threadIdx.y; // for d_B matrix

	float c_val = 0;

	if (row < rows_C && col < cols_C) // only want rows and columns that fit within the resultant matrix, otherwise, doing extra work
	{
		for (int i = 0; i < cols_A; i++) {
			if (row < rows_A && (i < rows_B && col < cols_B)) // explicitly check bounds
			{
				__syncthreads();
				c_val += d_A[row * cols_A + i] * d_B[i * cols_B + col];
				__syncthreads();
			}
		}
		d_C[row * cols_C + col] = c_val;
	}
}

template <typename T>
__global__ void CUDA_MAT_MULT_TILED(T *d_A, T *d_B, T *d_C, int rows_A, int cols_A, int rows_B, int cols_B, int rows_C, int cols_C, int TILE_WIDTH)
{

	int row = blockIdx.x * TILE_WIDTH + threadIdx.x; // for d_A matrix
	int col = blockIdx.y * TILE_WIDTH + threadIdx.y; // for d_B matrix

	int c_row = blockIdx.x * blockDim.x + threadIdx.x;
	int c_col = blockIdx.y * blockDim.y + threadIdx.y;

	T c_val = 0;

	if (row < rows_C && col < cols_C) // only want rows and columns that fit within the resultant matrix, otherwise, doing extra work
	{
		for (int i = 0; i < (cols_A + TILE_WIDTH - 1)/TILE_WIDTH; i++)
		{
			for (int j = 0; j < TILE_WIDTH; j++)
			{
				if ((i * TILE_WIDTH + j < cols_A && row < rows_A) && (i * TILE_WIDTH + j < rows_B && col < cols_B)) // don't go overbounds since d_A and d_B are not necessarily the same shape
				{
					c_val += d_A[row * cols_A + i * TILE_WIDTH + j] * d_B[(i * TILE_WIDTH + j) * cols_B + col];
				}
			}
		}

		d_C[(c_row * cols_C) + c_col] = c_val;
	}
}

template <typename T>
__global__ void CUDA_MAT_MULT_SHARED(T *d_A, T *d_B, T *d_C, int rows_A, int cols_A, int rows_B, int cols_B, int rows_C, int cols_C)
{

	int row = blockIdx.x * SHARED_TILE_WIDTH + threadIdx.x; // for d_A matrix
	int col = blockIdx.y * SHARED_TILE_WIDTH + threadIdx.y; // fpr d_B matrix

	int c_row = blockIdx.x * blockDim.x + threadIdx.x;
	int c_col = blockIdx.y * blockDim.y + threadIdx.y;

	T c_val = 0;

	__shared__ T s_A[SHARED_TILE_WIDTH][SHARED_TILE_WIDTH];
	__shared__ T s_B[SHARED_TILE_WIDTH][SHARED_TILE_WIDTH];

	for (int i = 0; i < (SHARED_TILE_WIDTH + cols_A - 1)/SHARED_TILE_WIDTH; i++) {

		s_A[threadIdx.x][threadIdx.y] = (i * SHARED_TILE_WIDTH + threadIdx.y < cols_A && row < rows_A) ? d_A[row * cols_A + i * SHARED_TILE_WIDTH + threadIdx.y] : 0.0;

		s_B[threadIdx.x][threadIdx.y] = (i * SHARED_TILE_WIDTH + threadIdx.x < rows_B && col < cols_B) ? d_B[(i * SHARED_TILE_WIDTH + threadIdx.x) * cols_B + col] : 0.0;

		__syncthreads();
		for (int j = 0; j < SHARED_TILE_WIDTH; j++)
		{
			c_val += s_A[threadIdx.x][j] * s_B[j][threadIdx.y];
		}
		__syncthreads();
	}

	if (row < rows_C && col < cols_C)
	{
		d_C[c_row * cols_C + c_col] = c_val;
	}
}