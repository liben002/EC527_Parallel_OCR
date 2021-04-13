cuda:
	nvcc -arch compute_35 -code sm_35 -std=c++11 -Xcompiler "-fopenmp" example_usage.cu -o example_usage.out
