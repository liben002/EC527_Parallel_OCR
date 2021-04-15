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
__global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int row_len, int col_len)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	float Pval = 0;

	for (int k = 0; k < row_len; k++) {
		Pval += Md[row*row_len+k] * Nd[k*row_len+col];
		__syncthreads();
	}

	Pd[row*row_len+col] = Pval;
}

