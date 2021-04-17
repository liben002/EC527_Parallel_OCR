#include "nn.hpp"
#include <iostream>
#include <time.h>
#include <cassert>

#define EPOCHS 100
#define START 600
#define END 1600
#define STEP_SIZE 100

/**
 * Function to test neural network
 * @returns none
 */
static void test(int row_length) {
	// Creating network with 3 layers for "iris.csv"
	// First layer neurons must match testing params
	machine_learning::neural_network::NeuralNetwork myNN = machine_learning::neural_network::NeuralNetwork({ {4, "none"}, {row_length, "relu"}, {row_length, "sigmoid"} });

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

	double duration_table[(END-START) / STEP_SIZE + 1][2];

	for (int i = START; i <= END; i+= STEP_SIZE)
	{
		duration_table[i/100 -1][0] = i;
		printf("Starting test with row_length of %d\n", i);
		// start the timer
		auto start = std::chrono::high_resolution_clock::now();  // Start clock

		test(i);

		// stop the timer
		auto stop = std::chrono::high_resolution_clock::now();  // Stopping the clock
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

		printf("Time for learning over %d epochs: %f seconds\n", EPOCHS, duration.count() / 1e6);
		duration_table[i/100 -1][1] = duration.count() / 1e6;
	}

	printf("ROW_LENGTH, TIME\n");
	for (int i = 0; i < (END-START) / STEP_SIZE + 1; i++)
	{
		printf("%.3f, %.3f\n", duration_table[i][0], duration_table[i][1]);
	}

	printf("DONE");

	return 0;
}
