cuda-v100:
	nvcc -arch compute_70 -code sm_70 -std=c++11 main.cu -o main.out