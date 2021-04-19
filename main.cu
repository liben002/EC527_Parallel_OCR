#include "nn.hpp"
#include <iostream>
#include <time.h>
#include <cassert>

#define EPOCHS 100
#define START 700
#define END 700
#define STEP_SIZE 100

/**
 * Function to test neural network
 * @returns none
 */
static void test(int row_length) {
	// Creating network with 3 layers for "iris.csv"
	// First layer neurons must match testing params
	machine_learning::neural_network::NeuralNetwork myNN = machine_learning::neural_network::NeuralNetwork({ {785, "none"}, {row_length, "relu"}, {row_length, "sigmoid"} });

	// Printing summary of model
	myNN.summary();

	// Training Model
	myNN.fit_from_csv("mnist.csv", true, EPOCHS, 0.3, false, 2, 32, true);

	printf("Testing predictions\n");
	// Testing predictions of model
	assert(machine_learning::argmax(myNN.single_predict({{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,253,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,252,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,173,253,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,253,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,113,253,244,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,233,252,122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,233,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,233,252,172,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,253,254,172,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,233,252,131,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,173,254,172,0,0,51,132,132,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,233,252,172,10,0,163,253,252,253,232,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,173,253,254,172,113,152,254,253,224,203,214,253,163,0,0,0,0,0,0,0,0,0,0,0,0,0,21,183,253,252,253,252,253,252,192,70,20,0,51,252,203,0,0,0,0,0,0,0,0,0,0,0,0,21,214,253,254,253,254,233,183,61,0,0,0,41,214,253,41,0,0,0,0,0,0,0,0,0,0,0,21,203,253,212,253,252,192,50,0,0,0,82,163,243,233,111,0,0,0,0,0,0,0,0,0,0,0,0,173,253,183,0,234,253,254,253,234,152,254,253,244,162,41,0,0,0,0,0,0,0,0,0,0,0,0,41,253,212,20,0,71,252,253,252,253,252,253,171,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,255,213,214,253,254,253,224,203,183,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,253,252,253,212,151,111,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6
}})) == 6);
	// assert(machine_learning::argmax( myNN.single_predict({{6.4, 2.9, 4.3, 1.3}})) == 1);
	// assert(machine_learning::argmax(myNN.single_predict({{6.2, 3.4, 5.4, 2.3}})) == 2);
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
