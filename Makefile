serial:
	g++ example_usage.cpp -std=c++11 -lrt -lm -o example_usage

parallel:
	g++ example_usage.cpp -std=c++11 -fopenmp -lrt -lm -o example_usage
