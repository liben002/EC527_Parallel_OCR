#include <iostream>
#include <vector>
#include <valarray>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <cuda.h>


// template <typename T>
// __global__ void CUDA_MAT_SUBT(T *d_A, T *d_B, T *d_C, int row_len, int col_len)
// {
// 	int row = blockIdx.x * blockDim.x + threadIdx.x;
// 	int col = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (row <= row_len && col <= col_len)
// 		d_C[row*col_len+col] = d_A[row*col_len+col] - d_B[row*col_len+col];

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

template <typename T>
std::pair<size_t, size_t> get_shape(const std::vector<std::valarray<T> > &A)
{
	const size_t sub_size = (*A.begin()).size();
	for (const auto &a : A)
	{
		// If supplied vector don't have same shape in all rows
		if (a.size() != sub_size)
		{
			std::cerr << "ERROR (" << __func__ << ") : " << "Supplied vector is not 2D Matrix" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	return std::make_pair(A.size(), sub_size);  // Return shape as pair
}

template <typename T>
std::vector<std::valarray<T> > operator-(const std::vector<std::valarray<T> > &A, const std::vector<std::valarray<T> > &B)
{
	const auto shape_a = get_shape(A);
	const auto shape_b = get_shape(B);

	printf("shape A: %d %d\n", shape_a.first, shape_a.second);
	printf("shape B: %d %d\n", shape_b.first, shape_b.second);
	// If vectors don't have equal shape
	if (shape_a.first != shape_b.first || shape_a.second != shape_b.second)
	{
		printf("BAD\n");
	}

	size_t mat_A_size = shape_a.first * shape_a.second * sizeof(T);
	size_t mat_B_size = shape_b.first * shape_b.second * sizeof(T);
	size_t mat_C_size = shape_a.first * shape_b.second * sizeof(T);
	printf("Matrix dimensions: %d x %d, %d x %d, %d x %d: %d\n", shape_a.first, shape_a.second, shape_b.first, shape_b.second, shape_a.first, shape_b.second);

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Allocate host memory
	printf("Allocating host vectors.\n");
	T *h_A = (T *) malloc(mat_A_size);
	T *h_B = (T *) malloc(mat_B_size);
	T *h_C = (T *) malloc(mat_C_size);

	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			h_A[i*shape_a.second + j] = A[i][j];
		}
	}

	for (int i = 0; i < shape_b.first; i++) {
		for (int j = 0; j < shape_b.second; j++) {
			h_B[i*shape_b.second + j] = B[i][j];
		}
	}

	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_b.second; j++) {
			h_C[i*shape_b.second + j] = 5;
		}
	}

	printf("h_A contains: \n");
	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			printf("%d ", h_A[i*shape_a.second + j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("h_B contains: \n");
	for (int i = 0; i < shape_b.first; i++) {
		for (int j = 0; j < shape_b.second; j++) {
			printf("%d ", h_B[i*shape_b.second + j]);
		}
		printf("\n");
	}

	// Allocate device vector
	printf("Allocating device vectors.\n");
	T *d_A = NULL;
	T *d_B = NULL;
	T *d_C = NULL;

	err = cudaMalloc((void **) &d_A, mat_A_size);
	err = cudaMalloc((void **) &d_B, mat_B_size);
	err = cudaMalloc((void **) &d_C, mat_C_size);

	printf ("Copying host vectors to CUDA device vectors\n");
	err = cudaMemcpy(d_A, h_A, mat_A_size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, h_B, mat_B_size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_C, h_C, mat_C_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(8, 8);
	dim3 dimGrid(4, 4);
	printf("Launching CUDA kernel with %d blocks and %d threads.\n", 4, 4 * 4);

	CUDA_MAT_MULT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, shape_a.first, shape_a.second, shape_b.first, shape_b.second, shape_a.first, shape_b.second);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch MMM kernel (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy output data from CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, mat_C_size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy matrix from device to host (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("h_C contains: \n");
	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_b.second; j++) {
			printf("%d ", h_C[i*shape_b.second + j]);
		}
		printf("\n");
	}

	std::vector<std::valarray<T> > C(shape_a.first);         // Vector to store result
	for (size_t i = 0; i < shape_a.first; i++) {  // For every row
		std::valarray<T> temp(1,shape_b.second);
		for (size_t j = 0; j < shape_b.second; j++) {
			temp[j] = h_C[i*shape_b.second + j];
		}
		C[i] = temp;            // Elementwise substraction
	}

	printf("Freeing\n");
	// Free device global memory
	err = cudaFree(d_A);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device matrix (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device matrix (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device matrix (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return C;  // Return new resultant 2D vector
}

int main() {

	std::vector<std::valarray<int> > A, B, C;

	for (int i = 0 ; i < 2; i++) {
		std::valarray<int> temp1(1,4);
		for (int j = 0; j < 4; j++) {
			temp1[j] = i+2;
		}
		A.push_back(temp1);
	}

	for (int i = 0 ; i < 1; i++) {
		std::valarray<int> temp1(1,2);
		for (int j = 0; j < 2; j++) {
			temp1[j] = i+3;
		}
		B.push_back(temp1);
	}

	C = A - B;

	printf("Correct value: \n");
	for (int i = 0 ; i < 2; i ++) {
		for (int j = 0; j < 4; j++) {
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}
}