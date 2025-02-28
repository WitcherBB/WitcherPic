#pragma once
#include <cstdint>

namespace witcher_pic {
	struct FilterInfo;
	auto hostDeviceInfo() -> void;
	auto hostMatFilter(uint8_t* source, float* model, const FilterInfo& info) -> uint8_t*;
	auto hostGrayCount(uint8_t* source, size_t size) -> size_t*;
	auto hostMapGrayImage(const uint8_t* source, const uint8_t* map_table, size_t size) -> void;
	auto hostTwoDimCrossCorre(const uint8_t* source, const float* model, unsigned s_w, unsigned s_h, unsigned m_w,
	                          unsigned m_h) -> uint8_t*;
	auto hostAddWeighted(size_t size, uint8_t* t, float w1, const uint8_t* s, float w2, uint8_t r) -> void;
	auto hostInsertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t*;
	auto hostGetEdgeInfo(int* dirmat, const uint8_t* source, const float* model_x, const float* model_y, unsigned s_width,
	                     unsigned s_height, unsigned m_width, unsigned m_height, bool l2_gradient) -> uint8_t*;
	auto hostNonMaxSuppression(const uint8_t* source, const int* dir, unsigned width, unsigned height) -> uint8_t*;
	auto hostTwoThreshold(const uint8_t* source, unsigned width, unsigned height, uint8_t l_threshold, uint8_t h_threshold) -> uint8_t*;
}
