#include "nn.hpp"
#include <iostream>
#include <time.h>
#include <cassert>

#define EPOCHS 100

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

/**
 * Function to test neural network
 * @returns none
 */
static void test() {
	// Creating network with 3 layers for "iris.csv"
	machine_learning::neural_network::NeuralNetwork myNN =
		machine_learning::neural_network::NeuralNetwork({
			{4, "none"},  // First layer with 3 neurons and "none" as activation
			{60,
			 "relu"},  // Second layer with 6 neurons and "relu" as activation
			{30, "sigmoid"}  // Third layer with 3 neurons and "sigmoid" as
							// activation
		});
	// Printing summary of model
	myNN.summary();
	// Training Model
	myNN.fit_from_csv("iris.csv", true, EPOCHS, 0.3, false, 2, 32, true);
	// Testing predictions of model
	assert(machine_learning::argmax(
			   myNN.single_predict({{5, 3.4, 1.6, 0.4}})) == 0);
	assert(machine_learning::argmax(
			   myNN.single_predict({{6.4, 2.9, 4.3, 1.3}})) == 1);
	assert(machine_learning::argmax(
			   myNN.single_predict({{6.2, 3.4, 5.4, 2.3}})) == 2);
	return;
}

/**
 * @brief Main function
 * @returns 0 on exit
 */
int main() {

	struct timespec time_start_CPU, time_end_CPU;

	// start the timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start_CPU);

	test();

	// stop the timer
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_end_CPU);

	printf("Time for learning over %d epochs: %f seconds\n", EPOCHS, interval(time_start_CPU, time_end_CPU));
	return 0;
}
