/*
 * AddVectors.cpp
 *
 *  Created on: Nov 2, 2021
 *      Author: nic
 */

#include <array>
#include <iostream>
#include <chrono>
#include <memory>

constexpr uint64_t DATASIZE = 1000000;

template <typename T>
void SumArraySequential(T& target, const T& a, const T& b)
{
	for (uint64_t i = 0; i < DATASIZE; i++)
	{
		target[i] = a[i] + b[i];
	}
}

int main()
{
	// Using a C-style array for now (with C++ smart-pointers)
	std::unique_ptr a = std::make_unique<int[]>(DATASIZE);
	std::unique_ptr b = std::make_unique<int[]>(DATASIZE);

	auto startTime = std::chrono::steady_clock::now();

	// Fill arrays with random data
	srand(42);
	for (uint64_t i = 0; i < DATASIZE; i++)
	{
		a[i] = rand();
		b[i] = rand();
	}

	auto dataFilledTime = std::chrono::steady_clock::now();

	// Sequentially sum arrays into target
	std::unique_ptr sc = std::make_unique<int[]>(DATASIZE);
	SumArraySequential(sc, a, b);

	auto sequentialCalculationTime = std::chrono::steady_clock::now();

	// Give output information
	std::cout << "Time to fill arrays with data: " << static_cast<std::chrono::duration<double>>(dataFilledTime - startTime).count() << std::endl;
	std::cout << "Time to sum arrays sequentially: " << static_cast<std::chrono::duration<double>>(sequentialCalculationTime - dataFilledTime).count() << std::endl;
}
