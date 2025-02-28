#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#ifndef PI
#define PI 3.1415926535
#endif
#ifndef PI_F
#define PI_F 3.141593F
#endif

namespace witcher_pic {
	__device__ __forceinline__ auto getModelSize(const float* model, size_t size) -> size_t;
	__device__ __forceinline__ auto twoDimConv(const uint8_t* source, const float* model, unsigned s_w, unsigned m_w,
	                                           unsigned m_h) -> float;
	__device__ __forceinline__ auto sort(uint8_t* mat, size_t size) -> void;
	__device__ __forceinline__ auto saturate(double x, double min_x = 0.0, double max_x = 1.0) -> double;
	__device__ __forceinline__ auto saturate(float x, float min_x = 0.0, float max_x = 1.0) -> float;
}

namespace witcher_pic {
	__device__ __forceinline__ auto getModelSize(const float* model, size_t size) -> size_t {
		size_t m_size = 0;
		for (size_t i = 0; i < size; ++i) {
			m_size += static_cast<size_t>(model[i] != 0.0F);
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

	__device__ __forceinline__ auto saturate(double x, double min_x, double max_x) -> double {
		return fmax(min_x, fmin(max_x, x));
	}

	__device__ __forceinline__ auto saturate(float x, float min_x, float max_x) -> float {
		return fmaxf(min_x, fminf(max_x, x));
	}
}
