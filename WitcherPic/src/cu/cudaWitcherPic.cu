#include "cudaWitcherPic.h"
#include "deviceLib.cuh"
#include "device_launch_parameters.h"
#include "witcherPic_types.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <variant>
#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#define CHECK_CUDA_ERR(errFunc, name) {\
			cudaError_t err = errFunc; \
			if (err != cudaSuccess) { \
				printf("CUDA Error (%s): %s\n", name, cudaGetErrorString(err)); \
			}\
		}
#define CHECK_CUDA_LAST_ERR(name) CHECK_CUDA_ERR(cudaGetLastError(), name)

namespace witcher_pic {
	namespace mutex {
		int* deviceMutex = nullptr;
	}

	__global__ auto cudaMatFilter(uint8_t* result, uint8_t* source, float* model, FilterInfo info) -> void {
		int cx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
		int cy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

		if (cx >= info.source_w || cy >= info.source_h) {
			return;
		}

		float color = 0;
		size_t ignored = 0;
		switch (info.type) {
		case CONV:
			for (auto rx = 0; rx < info.model_w; rx++) {
				for (auto ry = 0; ry < info.model_h; ry++) {
					const int xi = max(0, min(cx - info.rcx + rx, info.source_w - 1));
					const int yi = max(0, min(cy - info.rcy + ry, info.source_h - 1));
					color += model[ry * info.model_w + rx] * static_cast<float>(source[yi * info.source_w + xi]);
				}
			}
			break;
		case MEDIAN:
			size_t m_size = getModelSize(model, info.model_h * info.model_w);
			uint8_t* pixels = new uint8_t[m_size]{0};
			size_t idx = 0;

			for (auto rx = 0; rx < info.model_w; rx++) {
				for (auto ry = 0; ry < info.model_h; ry++) {
					if (model[ry * info.model_w + rx] != 0.0F) {
						const int xi = cx - info.rcx + rx;
						const int yi = cy - info.rcy + ry;
						if (xi < 0 || yi < 0 || xi >= info.source_w || yi >= info.source_h) {
							ignored += 1;
							continue;
						}
						auto ct = source[yi * info.source_w + xi];
						pixels[idx++] = ct;
					}
				}
			}

			sort(pixels, m_size - ignored);
			color = pixels[(size_t)((m_size - ignored - 1) / 2)];
		// 注意内存泄漏
			delete[] pixels;
			break;
		}
		result[cy * info.source_w + cx] = static_cast<uint8_t>(color);
	}

	__global__ auto cudaGrayCount(size_t* count_arr, uint8_t* source, size_t s_size) -> void {
		unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= s_size) {
			return;
		}
		atomicAdd(reinterpret_cast<unsigned long long*>(count_arr + source[idx]), 1);
	}

	__global__ auto cudaMapGrayImage(uint8_t* source, uint8_t* map_table) -> void {
		unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
		source[idx] = map_table[source[idx]];
	}

	__global__ auto cudaTwoDimCrossCorre(uint8_t* target, const uint8_t* source, const float* model,
	                                     unsigned s_w, unsigned s_h, unsigned m_w, unsigned m_h) -> void {
		unsigned center_x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned center_y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned m_center = (m_w - 1) / 2;

		// 边缘检测
		if (center_x < m_center || center_y < m_center || center_x >= s_w - m_center || center_y >= s_h - m_center) {
			return;
		}

		float color = 0;
		for (size_t idx = 0; idx < (size_t)(m_w * m_h); idx++) {
			unsigned rx = idx % m_w;
			unsigned ry = (unsigned)(idx / m_w);
			color += model[idx] * static_cast<float>(source[
				(center_y - m_center + ry) * s_w + (center_x - m_center + rx)]);
		}

		target[center_y * s_w + center_x] = static_cast<uint8_t>(__fmul_rz(
			__saturatef(__fdividef(fabsf(color), 255.0F)), 255.0F));
	}

	__global__ auto cudaAddWeighted(uint8_t* target, float w1, const uint8_t* other, float w2, uint8_t r,
	                                size_t size, bool l2_gradient) -> void {
		unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= size) {
			return;
		}
		uint8_t t_val = target[idx];
		uint8_t o_val = other[idx];
		target[idx] = static_cast<uint8_t>(saturate(
			l2_gradient ? sqrtf(w1 * t_val * t_val + w2 * o_val * o_val) : (w1 * t_val + w2 * o_val), 0, 255));
	}

	__global__ auto cudaInsertData(uint8_t* result, uint8_t** data, size_t datasize, int count) -> void {
		unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < datasize) {
			for (int i = 0; i < count; i++) {
				result[idx * count + i] = data[i][idx];
			}
		}
	}

	__global__ auto cudaGetAngleInfo(int* dir_data, const uint8_t* xdata, const uint8_t* ydata,
	                                 unsigned width, unsigned height) -> void {
		unsigned center_x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned center_y = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned idx = center_y * width + center_x;

		if (center_x >= width || center_y >= height) {
			return;
		}

		float x_val = xdata[idx];
		float y_val = ydata[idx];

		int angle = static_cast<int>(rintf(x_val == 0.0F ? 2 : atanf(y_val / x_val) / (PI_F / 4))) * 45;
		dir_data[idx] = angle == -90 ? 90 : (angle == -45 ? 135 : angle);
	}

	__global__ auto cudaNonMaxSuppression(uint8_t* result, const uint8_t* source, const int* dir, unsigned width,
	                                      unsigned height) -> void {
		unsigned c_x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned c_y = blockIdx.y * blockDim.y + threadIdx.y;
		size_t idx = c_y * width + c_x;
		if (c_x < 1 || c_x >= width - 1 || c_y < 1 || c_y >= height - 1) {
			return;
		}

		uint2 point1, point2;
		switch (dir[idx]) {
		case 0:
			point1 = {c_x - 1, c_y};
			point2 = {c_x + 1, c_y};
			break;
		case 45:
			point1 = {c_x - 1, c_y + 1};
			point2 = {c_x + 1, c_y - 1};
			break;
		case 90:
			point1 = {c_x, c_y + 1};
			point2 = {c_x, c_y - 1};
			break;
		case 135:
			point1 = {c_x + 1, c_y + 1};
			point2 = {c_x - 1, c_y - 1};
			break;
		default:
			return;
		}
		size_t idx1 = point1.y * width + point1.x;
		size_t idx2 = point2.y * width + point2.x;
		result[idx] = source[idx] >= source[idx1] && source[idx] >= source[idx2] ? source[idx] : 0;
	}

	__global__ auto cudaTwoThreshold(uint8_t* target, uint8_t* source, uint8_t l_threshold, uint8_t h_threshold,
	                                 unsigned width,
	                                 unsigned height) -> void {
		int2 center{(int)(blockIdx.x * blockDim.x + threadIdx.x), (int)(blockIdx.y * blockDim.y + threadIdx.y)};
		size_t c_idx = center.y * width + center.x;
		if (center.x >= (int)width || center.y >= (int)height) {
			return;
		}
		int2 neibor[8] = {
			int2{center.x + 1, center.y},
			int2{center.x, center.y - 1},
			int2{center.x - 1, center.y},
			int2{center.x, center.y + 1},

			int2{center.x + 1, center.y + 1},
			int2{center.x + 1, center.y - 1},
			int2{center.x - 1, center.y + 1},
			int2{center.x - 1, center.y - 1}
		};
		if (source[c_idx] > h_threshold) {
			target[c_idx] = source[c_idx];
		} else if (source[c_idx] <= l_threshold) {
			target[c_idx] = 0;
		} else {
			int i = 0;
			for (; i < 8; i++) {
				size_t idx = neibor[i].y * width + neibor[i].x;
				if (neibor[i].x < 0 || neibor[i].x >= (int)width || neibor[i].y < 0 || neibor[i].y >= (int)height) {
					continue;
				}
				if (source[idx] > h_threshold) {
					target[c_idx] = source[c_idx];
					break;
				}
			}
			if (i == 4) {
				target[c_idx] = 0;
			}
		}
	}

	__global__ auto cudaHoughTransform(size_t* houghzoom, const uint8_t* source, unsigned width, unsigned height,
	                                   unsigned houghsize, unsigned r_offset, double del_theta,
	                                   double del_radius) -> void {
		uint2 c_point{
			blockIdx.x * blockDim.x + threadIdx.x,
			blockIdx.y * blockDim.y + threadIdx.y
		};
		size_t idx = c_point.y * width + c_point.x;

		if (c_point.x >= width || c_point.y >= height || !source[idx]) {
			return;
		}
		const auto trans_func = [&](double theta) -> double {
			return (double)c_point.x * cos(theta) + (double)c_point.y * sin(theta);
		};

		for (unsigned i = 0; i < houghsize; i++) {
			double theta = i * del_theta;
			double radius = trans_func(theta);
			atomicAdd(reinterpret_cast<unsigned long long*>(&houghzoom[((int)floor(radius / del_radius) + (int)r_offset) * houghsize + i]), 1);
		}
	}

	__global__ auto cudaHoughPeak(size_t* peak, const size_t* houghzoom, unsigned houghsize) -> void {
		extern __shared__ size_t s_data[];
		unsigned tid = threadIdx.x;
		unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
		s_data[tid] = (idx < houghsize * houghsize) ? houghzoom[idx] : 0;
		__syncthreads();

		// for (unsigned i = 1; i <= blockDim.x / 2; i <<= 1) {
		// 	if (!(tid % (i << 1))) {
		// 		s_data[tid] = max(s_data[tid], s_data[tid + i]);
		// 	}
		// 	__syncthreads();
		// }

		// 2.
		for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				s_data[tid] = max(s_data[tid], s_data[tid + s]);
			}
			__syncthreads();
		}

		if (tid == 0) {
			atomicMax(reinterpret_cast<unsigned long long*>(peak), s_data[0]);
		}
	}

	__global__ auto cudaHoughLineCount(size_t* count, const size_t* houghzoom, size_t peak,
	                                   unsigned houghsize) -> void {
		unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

		if (idx < houghsize * houghsize && houghzoom[idx] == peak) {
			atomicAdd(reinterpret_cast<unsigned long long*>(count), 1);
		}
	}

	__global__ auto cudaHoughPeakLines(double* radius, double* thetas, const size_t* houghzoom,
	                                   size_t linesize, size_t peak, unsigned houghsize, double del_theta,
	                                   double del_radius, unsigned r_offset, size_t* index) -> void {
		uint2 point{
			blockDim.x * blockIdx.x + threadIdx.x,
			blockDim.y * blockIdx.y + threadIdx.y,
		};
		unsigned idx = point.y * houghsize + point.x;

		if (idx < houghsize * houghsize && houghzoom[idx] == peak) {
			size_t old_index = atomicAdd(reinterpret_cast<unsigned long long*>(index), 1);
			if (old_index < linesize) {
#ifdef _DEBUG
				printf("device: del_theta = %lf, del_radius = %lf, r_offset = %u\n", del_theta, del_radius, r_offset);
				printf("device: theta = %u, radius = %u\n", point.x, point.y);
#endif

				thetas[old_index] = (double)point.x * del_theta;
				radius[old_index] = ((double)point.y - (double)r_offset) * del_radius;
			}
		}
	}

	__global__ auto cudaTransform(uint8_t* result, const uint8_t* source, const int* trans_x, const int* trans_y,
	                              size_t size, unsigned result_width) -> void {
		unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx < size) {
			result[trans_y[idx] * result_width + trans_x[idx]] = source[idx];
		}
	}

#ifdef _DEBUG
	__global__ auto cudaTest() -> void {
		printf("x: %u, y: %u\n", threadIdx.x, threadIdx.y);
	}
#endif
	extern "C" {
		auto deviceInit() -> void {
			mutex::init();
		}

		auto hostDeviceInfo() -> void {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
			printf("Max registers per block: %d\n", prop.regsPerBlock);
			printf("Max shared memory per block: %llu\n", prop.sharedMemPerBlock);
		}

		auto hostMatFilter(uint8_t* source, float* model, const FilterInfo& info) -> uint8_t* {
			const size_t s_size = info.source_w * info.source_h;
			const size_t m_size = info.model_w * info.model_h;

			uint8_t* result = new uint8_t[s_size]{};
			uint8_t *sD, *rD;
			float* mD;

			cudaMalloc(&sD, s_size);
			cudaMalloc(&rD, s_size);
			cudaMalloc(&mD, m_size * sizeof(float));
			cudaMemcpy(sD, source, s_size, cudaMemcpyHostToDevice);
			cudaMemcpy(rD, result, s_size, cudaMemcpyHostToDevice);
			cudaMemcpy(mD, model, m_size * sizeof(float), cudaMemcpyHostToDevice);

			dim3 blockSize(32, 32);
			dim3 gridSize((static_cast<unsigned>(info.source_w) + blockSize.x - 1) / blockSize.x,
			              (static_cast<unsigned>(info.source_h) + blockSize.y - 1) / blockSize.y);

			cudaMatFilter<<<gridSize, blockSize>>>(rD, sD, mD, info);
			// cudaTest<<<1, 1024>>>();
			cudaDeviceSynchronize();
			cudaMemcpy(result, rD, s_size, cudaMemcpyDeviceToHost);

			cudaFree(sD);
			cudaFree(rD);
			cudaFree(mD);

			CHECK_CUDA_LAST_ERR("filter")

			return result;
		}

		auto hostGrayCount(uint8_t* source, size_t size) -> size_t* {
			size_t map_size = 256;
			uint8_t* sD;
			size_t* rD;
			auto result = new size_t[map_size]{};

			cudaMalloc(&sD, size);
			cudaMalloc(&rD, map_size * sizeof(size_t));
			cudaMemcpy(sD, source, size, cudaMemcpyHostToDevice);
			cudaMemcpy(rD, result, map_size * sizeof(size_t), cudaMemcpyHostToDevice);

			auto blockDimX = static_cast<unsigned>((size + 1023) / 1024);
			cudaGrayCount<<<blockDimX, 1024>>>(rD, sD, size);

			cudaMemcpy(result, rD, map_size * sizeof(size_t), cudaMemcpyDeviceToHost);

			CHECK_CUDA_ERR(cudaDeviceSynchronize(), "grayCount")
				cudaFree(sD);
				cudaFree(rD);

			CHECK_CUDA_LAST_ERR("grayCount")

			return result;
		}

		auto hostMapGrayImage(const uint8_t* source, const uint8_t* map_table, size_t size) -> void {
			const unsigned map_size = 256;
			uint8_t *cu_sData, *cu_map;
			cudaMalloc(&cu_sData, size);
			cudaMalloc(&cu_map, map_size);
			cudaMemcpy(cu_map, map_table, map_size, cudaMemcpyHostToDevice);
			cudaMemcpy(cu_sData, source, size, cudaMemcpyHostToDevice);

			auto blockDimX = static_cast<unsigned>((size + 1023) / 1024);
			cudaMapGrayImage<<<blockDimX, 1024>>>(cu_sData, cu_map);

			cudaFree(cu_sData);
			cudaFree(cu_map);
			CHECK_CUDA_LAST_ERR("mapGrayImage")
		}

		auto hostTwoDimCrossCorre(const uint8_t* source, const float* model, unsigned s_w, unsigned s_h, unsigned m_w,
		                          unsigned m_h) -> uint8_t* {
			const size_t s_size = s_w * s_h;
			const size_t m_size = m_w * m_h;
			uint8_t* result = new uint8_t[s_size]{0};
			uint8_t *sD, *rD;
			float* mD;
			cudaMalloc(&sD, s_size);
			cudaMalloc(&rD, s_size);
			cudaMalloc(&mD, m_size * sizeof(float));
			cudaMemcpy(sD, source, s_size, cudaMemcpyHostToDevice);
			cudaMemcpy(rD, result, s_size, cudaMemcpyHostToDevice);
			cudaMemcpy(mD, model, m_size * sizeof(float), cudaMemcpyHostToDevice);

			dim3 blockSize(32, 32);
			dim3 gridSize((s_w + blockSize.x - 1) / blockSize.x, (s_h + blockSize.y - 1) / blockSize.y);

			cudaTwoDimCrossCorre<<<gridSize, blockSize>>>(rD, sD, mD, s_w, s_h, m_w, m_h);

			cudaDeviceSynchronize();
			cudaMemcpy(result, rD, s_size, cudaMemcpyDeviceToHost);

			cudaFree(sD);
			cudaFree(mD);
			cudaFree(rD);
			CHECK_CUDA_LAST_ERR("twoDimCrossCorre")
			return result;
		}

		auto hostAddWeighted(size_t size, uint8_t* t, float w1, const uint8_t* s, float w2, uint8_t r) -> void {
			uint8_t *rD, *oD;
			cudaMalloc(&rD, size);
			cudaMalloc(&oD, size);
			cudaMemcpy(rD, t, size, cudaMemcpyHostToDevice);
			cudaMemcpy(oD, s, size, cudaMemcpyHostToDevice);

			unsigned gridSize = static_cast<unsigned>((size + 1023) / 1024);
			cudaAddWeighted<<<gridSize, 1024>>>(rD, w1, oD, w2, r, size, false);

			cudaMemcpy(t, rD, size, cudaMemcpyDeviceToHost);

			cudaFree(rD);
			cudaFree(oD);
			CHECK_CUDA_LAST_ERR("addWeighted")
		}

		auto hostInsertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t* {
			const auto result_size = count * datasize;
			auto result = new uint8_t[result_size]{};
			// host callable
			auto copy = new uint8_t*[count]{};

			uint8_t* cu_result;
			uint8_t** cu_data;
			cudaMalloc(&cu_data, count * sizeof(uint8_t*));
			cudaMalloc(&cu_result, result_size);
			for (int i = 0; i < count; i++) {
				if (datasize == 0 || data[i] == nullptr) {
					throw std::invalid_argument("Invalid datasize or data pointer.");
				}
				cudaMalloc(&copy[i], datasize);
				cudaMemcpy(copy[i], data[i], datasize, cudaMemcpyHostToDevice);
			}
			cudaMemcpy((void*)cu_data, (void*)copy, count * sizeof(uint8_t*), cudaMemcpyHostToDevice);
			cudaMemset(cu_result, 0, result_size);

			unsigned grid = static_cast<unsigned>((datasize + 1023) / 1024);
			cudaInsertData<<<grid, 1024>>>(cu_result, cu_data, datasize, count);
			cudaDeviceSynchronize();
			cudaMemcpy(result, cu_result, result_size, cudaMemcpyDeviceToHost);

			for (int i = 0; i < count; i++) {
				CHECK_CUDA_ERR(cudaFree(copy[i]), "copyFree")
			}
			cudaFree(cu_result);
			cudaFree((void*)cu_data);

			delete[] copy;
			CHECK_CUDA_LAST_ERR("insertData")
			return result;
		}

		auto hostGetEdgeInfo(int* dirmat, const uint8_t* source, const float* model_x, const float* model_y,
		                     unsigned s_width, unsigned s_height, unsigned m_width, unsigned m_height,
		                     bool l2_gradient) -> uint8_t* {
			size_t s_size = s_width * s_height;
			size_t m_size = m_width * m_height;

			uint8_t* result = new uint8_t[s_size];
			uint8_t *cu_source, *cu_xdata, *cu_ydata;
			float *cu_modelx, *cu_modely;
			int* cu_dirdata;

			cudaMalloc(&cu_source, s_size);
			cudaMalloc(&cu_xdata, s_size);
			cudaMalloc(&cu_ydata, s_size);
			cudaMalloc(&cu_modelx, m_size * sizeof(float));
			cudaMalloc(&cu_modely, m_size * sizeof(float));
			cudaMalloc(&cu_dirdata, s_size * sizeof(int));
			cudaMemcpy(cu_source, source, s_size, cudaMemcpyHostToDevice);
			cudaMemcpy(cu_modelx, model_x, m_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(cu_modely, model_y, m_size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemset(cu_xdata, 0, s_size);
			cudaMemset(cu_ydata, 0, s_size);
			cudaMemset(cu_dirdata, 0, s_size * sizeof(int));

			dim3 block(32, 32);
			dim3 grid((s_width + block.x - 1) / block.x, (s_height + block.y - 1) / block.y);
			cudaTwoDimCrossCorre<<<grid, block>>>(cu_xdata, cu_source, cu_modelx, s_width, s_height, m_width,
			                                      m_height);
			cudaTwoDimCrossCorre<<<grid, block>>>(cu_ydata, cu_source, cu_modely, s_width, s_height, m_width,
			                                      m_height);
			cudaDeviceSynchronize();
			cudaFree(cu_source);
			cudaFree(cu_modelx);
			cudaFree(cu_modely);

			cudaGetAngleInfo<<<grid, block>>>(cu_dirdata, cu_xdata, cu_ydata, s_width, s_height);
			cudaDeviceSynchronize();

			unsigned gridSize = static_cast<unsigned>((s_size + 1023) / 1024);
			cudaAddWeighted<<<gridSize, 1024>>>(cu_xdata, 0.5, cu_ydata, 0.5, 0, s_size, l2_gradient);
			cudaDeviceSynchronize();

			cudaFree(cu_ydata);
			cudaMemcpy(result, cu_xdata, s_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(dirmat, cu_dirdata, s_size * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(cu_dirdata);
			cudaFree(cu_xdata);

			CHECK_CUDA_LAST_ERR("sobelEdgeInfo")
			return result;
		}

		auto hostNonMaxSuppression(const uint8_t* source, const int* dir, unsigned width, unsigned height) -> uint8_t* {
			size_t size = width * height;
			uint8_t* result = new uint8_t[size]{};
			uint8_t *cu_source, *cu_result;
			int* cu_dir;
			cudaMalloc(&cu_source, size);
			cudaMalloc(&cu_dir, size * sizeof(int));
			cudaMalloc(&cu_result, size);
			cudaMemcpy(cu_source, source, size, cudaMemcpyHostToDevice);
			cudaMemcpy(cu_dir, dir, size * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(cu_result, result, size, cudaMemcpyHostToDevice);

			dim3 blockSize(32, 32);
			dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
			cudaNonMaxSuppression<<<gridSize, blockSize>>>(cu_result, cu_source, cu_dir, width, height);

			cudaMemcpy(result, cu_result, size, cudaMemcpyDeviceToHost);
			cudaFree(cu_source);
			cudaFree(cu_result);
			cudaFree(cu_dir);
			CHECK_CUDA_LAST_ERR("nonMaxSuppression")
			return result;
		}

		auto hostTwoThreshold(const uint8_t* source, unsigned width, unsigned height, uint8_t l_threshold,
		                      uint8_t h_threshold) -> uint8_t* {
			size_t size = width * height;
			uint8_t* result = new uint8_t[size]{};
			uint8_t *cu_source, *cu_result;
			cudaMalloc(&cu_source, size);
			cudaMalloc(&cu_result, size);
			cudaMemset(cu_result, 0, size);
			cudaMemcpy(cu_source, source, size, cudaMemcpyHostToDevice);

			dim3 block(32, 32);
			dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
			cudaTwoThreshold<<<grid, block>>>(cu_result, cu_source, l_threshold, h_threshold, width, height);

			cudaMemcpy(result, cu_result, size, cudaMemcpyDeviceToHost);
			cudaFree(cu_source);
			cudaFree(cu_result);
			CHECK_CUDA_LAST_ERR("twoThreshold")
			return result;
		}

		auto hostLineExtra(double** max_radius, double** max_thetas, size_t& line_size, const uint8_t* source,
		                   unsigned width, unsigned height, unsigned houghsize) -> void {
			// 如果houghsize是偶数，异常
			houghsize = (houghsize % 2) ? houghsize : houghsize + 1;

			const size_t s_size = width * height;
			unsigned r_offset = (houghsize - 1) / 2;
			double del_theta = PI / (double)houghsize;
			double del_radius = sqrt((double)(width * width + height * height)) / (double)r_offset;

#ifdef _DEBUG
		printf("host: del_theta = %lf, del_radius = %lf\n", del_theta, del_radius);
#endif

			size_t* cu_houghzoom;
			uint8_t* cu_source;
			cudaMalloc(&cu_houghzoom, houghsize * houghsize * sizeof(size_t));
			cudaMalloc(&cu_source, s_size);
			cudaMemcpy(cu_source, source, s_size, cudaMemcpyHostToDevice);
			cudaMemset(cu_houghzoom, 0, houghsize * houghsize * sizeof(size_t));
			// 霍夫变换，得到霍夫空间矩阵
			dim3 block(32, 32);
			dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
			unsigned blocksize = 1024;
			unsigned gridsize = (houghsize * houghsize + blocksize - 1) / blocksize;
			cudaHoughTransform<<<grid, block>>>(cu_houghzoom, cu_source, width, height, houghsize, r_offset, del_theta,
			                                    del_radius);
			cudaDeviceSynchronize();

			// 计算霍夫空间投票数最多的点
			thrust::device_ptr<size_t> hough_ptr(cu_houghzoom);
			auto hough_last_ptr = hough_ptr + houghsize * houghsize;

			double *cu_radius, *cu_thetas;
			size_t* cu_index;

			// // 1. 计算投票峰值
			size_t peak = thrust::reduce(thrust::device, hough_ptr, hough_last_ptr, (size_t)0,
			                             thrust::maximum<size_t>());
			// // 2. 计算等于峰值的点的数量
			line_size = thrust::count(thrust::device, hough_ptr, hough_last_ptr, peak);
			// 3. 根据峰值点的数量分配内存，获取峰值点的集合
			size_t line_bytes = line_size * sizeof(double);
			*max_radius = new double[line_size];
			*max_thetas = new double[line_size];
			cudaMalloc(&cu_radius, line_bytes);
			cudaMalloc(&cu_thetas, line_bytes);
			cudaMalloc(&cu_index, sizeof(size_t));
			cudaMemset(cu_index, 0, sizeof(size_t));
			grid = {(houghsize + block.x - 1) / block.x, (houghsize + block.y - 1) / block.y};
			cudaHoughPeakLines<<<grid, block>>>(cu_radius, cu_thetas, cu_houghzoom, line_size, peak, houghsize,
			                                    del_theta, del_radius, r_offset, cu_index);
			cudaDeviceSynchronize();

			CHECK_CUDA_ERR(cudaMemcpy(*max_radius, cu_radius, line_bytes, cudaMemcpyDeviceToHost), "radius cpy")
			CHECK_CUDA_ERR(cudaMemcpy(*max_thetas, cu_thetas, line_bytes, cudaMemcpyDeviceToHost), "thetas cpy")

				cudaFree(cu_radius);
				cudaFree(cu_thetas);
				cudaFree(cu_index);

				cudaFree(cu_source);
				cudaFree(cu_houghzoom);
			CHECK_CUDA_LAST_ERR("lineExtra")
		}

		auto hostDrawLine(uint8_t* source, unsigned width, unsigned height, double radius, double theta,
		                  uint8_t brightness, int thickness) -> void {
			size_t size = width * height;
			const double costheta = cos(theta);
			const double sintheta = sin(theta);
			const double cottheta = costheta / sintheta;
			int lower = thickness / 2;
			int higher = thickness - lower - 1;

			thrust::counting_iterator<size_t> idx_it;
			thrust::device_vector<size_t> dv_idx_it(idx_it, idx_it + size);
			// thrust::host_vector<uint8_t> hv_source(source, source + size);
			thrust::device_vector<uint8_t> dv_source(source, source + size);

			thrust::device_vector<bool> line_model(size);
			thrust::transform(thrust::device, dv_idx_it.begin(), dv_idx_it.end(), line_model.begin(),
			                  [width, theta, radius, lower, higher, sintheta, costheta, cottheta] __device__ (
			                  const size_t& idx) ->
			                  uint8_t {
				                  int x = idx % width;
				                  int y = idx / width;
				                  int cal;
				                  bool is_horizontal = theta <= PI / 4 || theta >= PI * 3 / 4;
				                  return is_horizontal
					                         ? (cal = (int)round(-tan(theta) * y + radius / costheta), (
						                            x >= cal - lower))
					                           && (x <= cal + higher)
					                         : (cal = (int)round(-cottheta * x + radius / sintheta), (y >= cal - lower))
					                           &&
					                           (y <= cal + higher);
			                  });

			thrust::transform_if(thrust::device, dv_source.begin(), dv_source.end(), line_model.begin(),
			                     dv_source.begin(),
			                     [brightness] __device__ (const uint8_t&) -> uint8_t {
				                     return brightness;
			                     }, thrust::identity<bool>());

			thrust::copy(dv_source.begin(), dv_source.end(), source);
		}

		auto hostRotate(const uint8_t* source, double theta, unsigned width, unsigned height, unsigned& new_width,
		                unsigned& new_height) -> uint8_t* {
			const size_t size = width * height;
			thrust::device_vector<int> x_mat(size), y_mat(size);
			thrust::device_vector<int> x_result(size), y_result(size);
			thrust::counting_iterator<size_t> counting_it;

			thrust::transform(thrust::device, counting_it, counting_it + size, x_mat.begin(),
			                  [width] __device__ (const size_t& idx) -> int {
				                  return idx % width;
			                  });
			thrust::transform(thrust::device, counting_it, counting_it + size, y_mat.begin(),
			                  [width] __device__ (const size_t& idx) -> int {
				                  return idx / width;
			                  });
			thrust::transform(thrust::device, x_mat.begin(), x_mat.end(), y_mat.begin(), x_result.begin(),
			                  [theta] __device__ (const int& x, const int& y) -> int {
				                  return (int)round(y * sin(theta) + x * cos(theta));
			                  });
			thrust::transform(thrust::device, x_mat.begin(), x_mat.end(), y_mat.begin(), y_result.begin(),
			                  [theta] __device__ (const int& x, const int& y) -> int {
				                  return (int)round(y * cos(theta) - x * sin(theta));
			                  });
			struct _MaxMin {
				int max;
				int min;
			} x_mm, y_mm;
			x_mm.min = thrust::reduce(thrust::device, x_result.begin(), x_result.end(), INT32_MAX, thrust::minimum<int>());
			y_mm.min = thrust::reduce(thrust::device, y_result.begin(), y_result.end(), INT32_MIN, thrust::minimum<int>());
			thrust::transform(thrust::device, x_result.begin(), x_result.end(),
			                  thrust::constant_iterator<int>(x_mm.min),
			                  x_result.begin(), thrust::minus<int>());
			thrust::transform(thrust::device, y_result.begin(), y_result.end(),
			                  thrust::constant_iterator<int>(y_mm.min),
			                  y_result.begin(), thrust::minus<int>());
			x_mm.max = thrust::reduce(thrust::device, x_result.begin(), x_result.end(), INT32_MAX, thrust::maximum<int>());
			y_mm.max = thrust::reduce(thrust::device, y_result.begin(), y_result.end(), INT32_MIN, thrust::maximum<int>());
			new_width = x_mm.max + 1;
			new_height = y_mm.max + 1;
			const size_t newsize = new_width * new_height;
			uint8_t* result = new uint8_t[newsize];
			uint8_t *cu_result, *cu_source;
			cudaMalloc(&cu_result, newsize);
			cudaMalloc(&cu_source, size);
			cudaMemset(cu_result, 0, newsize);
			cudaMemcpy(cu_source, source, size, cudaMemcpyHostToDevice);
			unsigned blocksize(1024);
			unsigned gridsize((newsize + blocksize - 1) / blocksize);
			cudaTransform<<<gridsize, blocksize>>>(cu_result, cu_source, x_result.data().get(), y_result.data().get(),
			                                       size, new_width);
			cudaDeviceSynchronize();
			thrust::transform(thrust::device, counting_it, counting_it + newsize,
			                  thrust::device_pointer_cast(cu_result),
			                  [cu_result, new_width, new_height] __device__ (const size_t& idx) -> uint8_t {
				                  unsigned x = idx % new_width;
				                  unsigned y = idx / new_width;
				                  if (x > 0 && x < new_width - 1 && y > 0 && y < new_height - 1 && !cu_result[idx] &&
				                      cu_result[idx - new_width] && cu_result[idx + 1] &&
				                      cu_result[idx - 1] && cu_result[idx + new_width]) {
					                  cu_result[idx] =
					                  (cu_result[idx - new_width] + cu_result[idx + 1] +
					                   cu_result[idx - 1] + cu_result[idx + new_width]) / 4;
				                  }
				                  return cu_result[idx];
			                  });
			cudaMemcpy(result, cu_result, newsize, cudaMemcpyDeviceToHost);
			cudaFree(cu_result);
			cudaFree(cu_source);
			CHECK_CUDA_LAST_ERR("rotate")
			return result;
		}

		auto hostCalc(Operator op, Number* data1, Number* data2, ptrdiff_t data_size, size_t type_size,
		              bool is_floating, bool is_unsigned) -> Number* {
			auto result = new Number[data_size]{};

			thrust::device_vector<Number> v1(data1, data1 + data_size);
			thrust::device_vector<Number> v2(data2, data2 + data_size);
			thrust::device_vector<Number> cu_result(data_size);

			thrust::transform(thrust::device, v1.begin(), v1.end(), v2.begin(), cu_result.begin(),
			                  [type_size, is_floating, is_unsigned] __device__ (
			                  const Number& n1, const Number& n2) -> Number {
				                  return Number();
			                  });
			thrust::copy(cu_result.begin(), cu_result.end(), result);
			return result;
		}
	}
}
