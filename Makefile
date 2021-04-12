serial:
	g++ example_usage.cpp -lrt -lm -o example_usage

parallel:
	g++ example_usage.cpp -fopenmp -lrt -lm -o example_usage
