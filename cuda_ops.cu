#include <cuda_runtime_api.h>
#include <cuda.h>

template <typename T>
__global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int A_rows, int A_cols, int B_rows, int B_cols, int C_rows, int C_cols) {

	T c_val = 0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < (A_cols); k++) {
			if ((k < A_cols && row < A_rows) && (k < B_rows && col < B_cols))
				c_val += d_A[row * A_cols + k] * d_B[k * B_cols + col];

	}

	if (row < C_rows && col < C_cols){
		d_C[((blockIdx.y * blockDim.y + threadIdx.y) * C_cols) + (blockIdx.x * blockDim.x) + threadIdx.x] = c_val;
	}
}
