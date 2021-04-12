#include <iostream>
#include <vector>
#include <valarray>
#include <iomanip>

__global__ void CUDA_MAT_SUBT(int *d_A, int *d_B, int *d_C)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// if (row < 4 && col < 4) {
		// for (int k = 0; k < 4; k++) {
			d_C[row*4+col] = d_A[row*4+col] - d_B[row*4+col];
			// __syncthreads();
		// }
	// }
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

	printf("shape: %d %d\n", shape_a.first, shape_a.second);
	// If vectors don't have equal shape
	if (shape_a.first != shape_b.first || shape_a.second != shape_b.second)
	{
		printf("BAD\n");
	}

	size_t mat_size = shape_a.first * shape_a.second * sizeof(int);
	printf("Matrix dimensions: %d x %d, Size of matrix in bytes: %d\n", shape_a.first, shape_a.second, mat_size);

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Allocate host memory
	printf("Allocating host vectors.\n");
	T *h_A = (T *) malloc(mat_size);
	T *h_B = (T *) malloc(mat_size);
	T *h_C = (T *) malloc(mat_size);

	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			h_A[i*shape_a.first + j] = A[i][j];
		}
	}

	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			h_B[i*shape_a.first + j] = B[i][j];
		}
	}

	printf("h_A contains: \n");
	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			printf("%d ", h_A[i*shape_a.first + j]);
		}
		printf("\n");
	}
	printf("\n");

	printf("h_B contains: \n");
	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			printf("%d ", h_B[i*shape_a.first + j]);
		}
		printf("\n");
	}

	// Allocate device vector
	printf("Allocating device vectors.\n");
	int *d_A = NULL;
	int *d_B = NULL;
	int *d_C = NULL;

	err = cudaMalloc((void **) &d_A, mat_size);
	err = cudaMalloc((void **) &d_B, mat_size);
	err = cudaMalloc((void **) &d_C, mat_size);

	printf ("Copying host vectors to CUDA device vectors\n");
	err = cudaMemcpy(d_A, h_A, mat_size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, h_B, mat_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(4, 4);
	dim3 dimGrid(1, 1);
	printf("Launching CUDA kernel with %d blocks and %d threads.\n", 4, 4 * 4);

	CUDA_MAT_SUBT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch MMM kernel (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy output data from CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, mat_size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy matrix from device to host (error code: %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("h_C contains: \n");
	for (int i = 0; i < shape_a.first; i++) {
		for (int j = 0; j < shape_a.second; j++) {
			printf("%d ", h_C[i*shape_a.first + j]);
		}
		printf("\n");
	}

	printf("Freeing\n");
	// Free device global memory
	err = cudaFree(d_A);

	err = cudaFree(d_B);

	err = cudaFree(d_C);

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

	std::vector<std::valarray<T> > C;         // Vector to store result
	for (size_t i = 0; i < A.size(); i++) {  // For every row
		C.push_back(A[i] - B[i]);            // Elementwise substraction
	}

	return C;  // Return new resultant 2D vector
}

// void print(const char* rem, const std::valarray<int>& v)
// {
//     std::cout << std::left << std::setw(36) << rem << std::right;
//     for (int n: v) std::cout << std::setw(3) << n;
//     std::cout << '\n';
// }

int main() {

	std::vector<std::valarray<int> > A, B, C;

	for (int i = 0 ; i < 4; i++) {
		std::valarray<int> temp1(1,4);
		std::valarray<int> temp2(1,4);
		for (int i = 0; i < 4; i++) {
			temp1[i] = i;
			temp2[i] = i*4;
		}
		A.push_back(temp1);
		B.push_back(temp2);
	}

	C = A - B;

	printf("Correct value: \n");
	for (int i = 0 ; i < 4; i ++) {
		for (int j = 0; j < 4; j++) {
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}
}