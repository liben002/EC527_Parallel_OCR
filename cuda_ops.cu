#include <cuda_runtime_api.h>
#include <cuda.h>

template <typename T>
__global__ void CUDA_MAT_SUBT(T *d_A, T *d_B, T *d_C, int row_len, int col_len)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < row_len && col < col_len)
		d_C[row*col_len+col] = d_A[row*col_len+col] - d_B[row*col_len+col];
}

template <typename T>
__global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int row_len_dA, int col_len_dA, int row_len_dB, int col_len_dB)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (!(row >= row_len_dA || col >= col_len_dA || row >= row_len_dB || col >= col_len_dB))
	{
		float Pval = 0;
		for (int i = 0; i < row_len_dA; i++) {
			for (int j = 0; j < col_len_dB; j++) {
			Pval += d_A[row*row_len_dA+i] * d_B[j*row_len_dB+col];
			__syncthreads();
		}
		d_C[row*row_len_dB+col] = Pval;
	}
}

