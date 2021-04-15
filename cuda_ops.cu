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

// template <typename T>
// __global__ void CUDA_MAT_MULT(T *d_A, T *d_B, T *d_C, int row_len_dA, int col_len_dA, int row_len_dB, int col_len_dB)
// {
// 	int row = blockIdx.x * blockDim.x + threadIdx.x;
// 	int col = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (!(row >= row_len_dA || col >= col_len_dA || row >= row_len_dB || col >= col_len_dB))
// 	{
// 		float Pval = 0;
// 		for (int i = 0; i < row_len_dA; i++) {
// 			for (int j = 0; j < col_len_dB; j++) {
// 			Pval += d_A[row*row_len_dA+i] * d_B[j*row_len_dB+col];
// 			__syncthreads();
// 			}
// 		}
// 		d_C[row*row_len_dB+col] = Pval;
// 	}
// }

template <typename T>
__global__ void CUDA_MAT_MULT(T* A, T* B, T* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y + threadIdx.y;
    int Col = blockIdx.x + threadIdx.x;

    for (int k = 0; k < (ACols - 1); k++) {
            if ((k < ACols && Row < ARows) && (k < BRows && Col < BCols))
                CValue += A[Row*ACols + k] * B[(k)*BCols + Col];

    }

    if (Row < CRows && Col < CCols){
    	C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
    }
}

