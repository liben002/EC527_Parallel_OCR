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


__global__ void CUDA_MAT_MULT_TILED(T* A, T* B, T* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols, int TILE_DIM) {

	T CValue = 0;

	int Row = blockIdx.y*TILE_DIM + threadIdx.y;
	int Col = blockIdx.x*TILE_DIM + threadIdx.x;

	for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

		for (int n = 0; n < TILE_DIM; ++n) 
			if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
				CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];

	}

	if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}