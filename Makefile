all: serial parallel

serial:
	g++ main.cpp -O3 -std=c++11 -lrt -lm -o serial

parallel:
	g++ main.cpp -O3 -std=c++11 -fopenmp -lrt -lm -o parallel

clean:
	rm serial parallel
