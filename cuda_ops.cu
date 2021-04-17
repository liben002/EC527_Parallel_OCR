#include <cuda_runtime_api.h>
#include <cuda.h>

template <typename T>
__global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols) {

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

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols, int TILE_WIDTH)
{
	float CValue = 0;

	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	for (int k = 0; k < (TILE_WIDTH + ACols - 1)/TILE_WIDTH; k++) {
		if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows)
		{
			As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_WIDTH + threadIdx.x];
		}
		else
		{
			As[threadIdx.y][threadIdx.x] = 0.0;
		}

		if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols)
		{
			Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
		}
		else
		{
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		}

		__syncthreads();

		for (int n = 0; n < TILE_WIDTH; ++n)
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

		__syncthreads();
	}

	if (Row < CRows && Col < CCols)
	{
		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) + (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
	}
}