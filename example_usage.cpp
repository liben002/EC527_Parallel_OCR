#include "nn.hpp"
#include <iostream>
#include <time.h>
#include <cassert>
#include <omp.h>

#define EPOCHS 100
#define THREADS 4

int clock_gettime(clockid_t clk_id, struct timespec *tp);

double interval(struct timespec start, struct timespec end)
{
	struct timespec temp;
	temp.tv_sec = end.tv_sec - start.tv_sec;
	temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	if (temp.tv_nsec < 0)
	{
		temp.tv_sec = temp.tv_sec - 1;
		temp.tv_nsec = temp.tv_nsec + 1000000000;
	}
	return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

void detect_threads_setting()
{
	long int i, ognt;
	char * env_ONT;

	//Find out how many threads OpenMP thinks it is wants to use
	#pragma omp parallel for
	for (i=0; i<1; i++) {
		ognt = omp_get_num_threads();
	}

	printf("omp's default number of threads is %d\n", ognt);

	//If this is illegal (0 or less), default to the "#define THREADS" value that is defined above
	if (ognt <= 0) {
		if (THREADS != ognt) {
			printf("Overriding with #define THREADS value %d\n", THREADS);
			ognt = THREADS;
		}
	}

	omp_set_num_threads(ognt);

	// Once again ask OpenMP how many threads it is going to use
	#pragma omp parallel for
	for (i=0; i<1; i++) {
		ognt = omp_get_num_threads();
	}

	printf("Using %d threads for OpenMP\n", ognt);
}

/**
 * Function to test neural network
 * @returns none
 */
static void test() {
	// Creating network with 3 layers for "iris.csv"
	// First layer neurons must match testing params
	machine_learning::neural_network::NeuralNetwork myNN = machine_learning::neural_network::NeuralNetwork({ {4, "none"}, {600, "relu"}, {300, "sigmoid"} });

	// Printing summary of model
	myNN.summary();

	// Training Model
	myNN.fit_from_csv("iris.csv", true, EPOCHS, 0.3, false, 2, 32, true);

	printf("Testing predictions\n");
	// Testing predictions of model
	assert(machine_learning::argmax(myNN.single_predict({{5, 3.4, 1.6, 0.4}})) == 0);
	assert(machine_learning::argmax( myNN.single_predict({{6.4, 2.9, 4.3, 1.3}})) == 1);
	assert(machine_learning::argmax(myNN.single_predict({{6.2, 3.4, 5.4, 2.3}})) == 2);
	return;
}

/**
 * @brief Main function
 * @returns 0 on exit
 */
int main() {

	struct timespec time_start_CPU, time_end_CPU;

	detect_threads_setting();

	// start the timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start_CPU);

	test();

	// stop the timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_end_CPU);

	printf("Time for learning over %d epochs: %f seconds\n", EPOCHS, interval(time_start_CPU, time_end_CPU));
	return 0;
}
