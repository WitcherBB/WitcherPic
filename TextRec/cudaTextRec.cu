#include "cudaTextRec.cuh"
#include "deviceLib.cuh"
#include "textRec_types.h"

#include <cstdio>
#include <functional>

#define CHECK_CUDA_ERR(errFunc, name) {\
			cudaError_t err = ##errFunc##; \
			if (err != cudaSuccess) { \
				printf("CUDA Error (%s): %s\n", ##name##, cudaGetErrorString(err)); \
			}\
		}

namespace witcher_pic {
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
					const int xi = cx - info.rcx + rx;
					const int yi = cy - info.rcy + ry;
					if (xi < 0 || yi < 0 || xi >= info.source_w || yi >= info.source_h) {
						continue;
					}
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
		// ×¢ÒâÄÚ´æÐ¹Â©
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
		atomicAdd(count_arr + source[idx], 1);
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

		// ±ßÔµ¼ì²â
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

		target[center_y * s_w + center_x] = static_cast<uint8_t>(saturate(abs(color), 0.0F, 255.0F));
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
		if (idx >= datasize) {
			return;
		}
		for (int i = 0; i < count; i++) {
			result[idx * count + i] = data[i][idx];
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

		int angle = static_cast<int>(round(x_val == 0.0F ? 2 : atanf(y_val / x_val) / (PI_F / 4))) * 45;
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
			for (; i < 4; i++) {
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

#ifdef _DEBUG
	__global__ auto cudaTest() -> void {
		printf("x: %u, y: %u\n", threadIdx.x, threadIdx.y);
	}
#endif

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

		CHECK_CUDA_ERR(cudaGetLastError(), "filter")

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

		CHECK_CUDA_ERR(cudaGetLastError(), "grayCount")

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
		CHECK_CUDA_ERR(cudaGetLastError(), "mapGrayImage")
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
		CHECK_CUDA_ERR(cudaGetLastError(), "twoDimCrossCorre")
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
		CHECK_CUDA_ERR(cudaGetLastError(), "addWeighted")
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
			uint8_t*& cu_data_i = copy[i];
			cudaMalloc(&cu_data_i, datasize);
			cudaMemcpy(cu_data_i, data[i], datasize, cudaMemcpyHostToDevice);
		}
		cudaMemcpy(cu_data, copy, count * sizeof(uint8_t*), cudaMemcpyHostToDevice);
		cudaMemcpy(cu_result, result, result_size, cudaMemcpyHostToDevice);

		unsigned grid = static_cast<unsigned>((datasize + 1023) / 1024);
		cudaInsertData<<<grid, 1024>>>(cu_result, cu_data, datasize, count);

		cudaMemcpy(result, cu_result, result_size, cudaMemcpyDeviceToHost);

		for (int i = 0; i < count; i++) {
			cudaError_t err = cudaFree(copy[i]);
			if (err != cudaSuccess) {
				printf("CUDA Error (copyFree): %s\n", cudaGetErrorString(err));
			}
		}
		cudaFree(cu_result);
		cudaFree(cu_data);

		delete[] copy;
		CHECK_CUDA_ERR(cudaGetLastError(), "insertData")
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
		cudaAddWeighted<<<gridSize, 1024>>>(cu_xdata, 0.5, cu_ydata, 0.5, 0, s_size, false);
		cudaDeviceSynchronize();

		cudaFree(cu_ydata);
		cudaMemcpy(result, cu_xdata, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(dirmat, cu_dirdata, s_size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(cu_dirdata);
		cudaFree(cu_xdata);
		
		CHECK_CUDA_ERR(cudaGetLastError(), "sobelEdgeInfo")
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
		CHECK_CUDA_ERR(cudaGetLastError(), "nonMaxSuppression")
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
		CHECK_CUDA_ERR(cudaGetLastError(), "twoThreshold1")
			cudaFree(cu_source);
			cudaFree(cu_result);
		CHECK_CUDA_ERR(cudaGetLastError(), "twoThreshold2")
		return result;
	}
}
