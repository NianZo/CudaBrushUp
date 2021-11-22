/*
 * AddVectors.cpp
 *
 *  Created on: Nov 2, 2021
 *      Author: nic
 */

#include <array>
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>
#include <cuda.h>

constexpr uint64_t DATASIZE = 100000000;

template <typename T>
class GPUMemory
{
public:
	GPUMemory(size_t allocSize)
	{
		cudaMallocManaged(&gpuPtr, allocSize);
	}

	GPUMemory(std::vector<T> hostData)
	{
		cudaMallocManaged(&gpuPtr, hostData.size() * sizeof(T));
		std::copy(hostData.begin(), hostData.end(), gpuPtr);
	}

	~GPUMemory()
	{
		cudaFree(gpuPtr);
	}

	__device__ T operator[](uint64_t index) const
	{
		return gpuPtr[index];
	}

	__device__ T& operator[](uint64_t index)
	{
		return gpuPtr[index];
	}
private:
	T* gpuPtr;
};

template <typename T>
void SumArraySequential(T& target, const T& a, const T& b)
{
	for (uint64_t i = 0; i < DATASIZE; i++)
	{
		target[i] = a[i] + b[i];
	}
}


__global__ void SumArrayCuda(GPUMemory<int>& target, const GPUMemory<int>& a, const GPUMemory<int>& b)
{
	uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < DATASIZE)
	{
		target[idx] = a[idx] + b[idx];
	}
}

int main()
{
	auto startTime = std::chrono::steady_clock::now();

	// Fill arrays with random data
	std::vector<int> hostA(DATASIZE);
	std::vector<int> hostB(DATASIZE);
	srand(42);
	for (uint64_t i = 0; i < DATASIZE; i++)
	{
		hostA[i] = rand();
		hostB[i] = rand();
	}
	GPUMemory<int> a(hostA);
	GPUMemory<int> b(hostB);

	auto dataFilledTime = std::chrono::steady_clock::now();

	// Offload calculation to GPU
	GPUMemory<int> sp(DATASIZE * sizeof(int));
	uint64_t blocks = DATASIZE / 32;
	SumArrayCuda<<<blocks, 32>>>(sp, a, b);
	cudaDeviceSynchronize();

	auto cudaCalculationTime = std::chrono::steady_clock::now();


	// Give output information
	std::cout << "Time to fill arrays with data: " << static_cast<std::chrono::duration<double>>(dataFilledTime - startTime).count() << std::endl;
	std::cout << "Time to sum arrays on GPU: " << static_cast<std::chrono::duration<double>>(cudaCalculationTime - dataFilledTime).count() << std::endl;
}
