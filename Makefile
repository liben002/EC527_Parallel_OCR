all: serial parallel avx

serial:
	g++ main.cpp -O3 -std=c++11 -lrt -lm -o serial

parallel:
	g++ main.cpp -O3 -std=c++11 -fopenmp -lrt -lm -o parallel

avx:
	icpc main.cpp -use-intel-optimized-headers -O3 -std=c++11 -openmp -lrt -lm -o avx 

clean:
	rm serial parallel avx
