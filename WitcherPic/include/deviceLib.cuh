#pragma once
#include <cuda_runtime.h>
#include <numbers>
#include "device_launch_parameters.h"

#ifndef PI
#define PI std::numbers::pi_v<double>
#endif
#ifndef PI_F
#define PI_F std::numbers::pi_v<float>
#endif

#define LOCK 1
#define UNLOCK 0

namespace witcher_pic {
	__device__ __forceinline__ auto getModelSize(const float* model, size_t size) -> size_t;
	__device__ __forceinline__ auto sort(uint8_t* mat, size_t size) -> void;

	namespace mutex {
		__device__ __forceinline__ auto lock(int* pmutex) -> void;
		__device__ __forceinline__ auto unlock(int* pmutex) -> void;
	}}

namespace witcher_pic {
	__device__ __forceinline__ auto getModelSize(const float* model, size_t size) -> size_t {
		size_t m_size = 0;
		for (size_t i = 0; i < size; ++i) {
			m_size += (model[i] != 0.0F);
		}
		return m_size;
	}

	__device__ __forceinline__ auto sort(uint8_t* mat, size_t size) -> void {
		for (size_t i = 0; i < size - 1; ++i) {
			for (size_t j = i + 1; j < size; ++j) {
				if (mat[i] > mat[j]) {
					auto t = mat[i];
					mat[i] = mat[j];
					mat[j] = t;
				}
			}
		}
	}
}

namespace witcher_pic::mutex {
	__device__ __forceinline__ auto lock(int* pmutex) -> void {
		while (atomicCAS(pmutex, UNLOCK, LOCK) != LOCK) {
		}
	}

	__device__ __forceinline__ auto unlock(int* pmutex) -> void {
		atomicExch(pmutex, UNLOCK);
	}
}
