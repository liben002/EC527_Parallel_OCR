#include <cuda_runtime_api.h>
#include <cuda.h>

template <typename T>
__global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols) {

	T c_val = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < A_cols; i++) {
			if ((i < A_cols && row < A_rows) && (i < B_rows && col < B_cols))
				c_val += d_A[row * A_cols + i] * d_B[i * B_cols + col];

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