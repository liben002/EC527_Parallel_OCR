serial:
	g++ example_usage.cpp -O3 -std=c++11 -lrt -lm -o example_usage

parallel:
	g++ example_usage.cpp -O3 -std=c++11 -fopenmp -lrt -lm -o example_usage
