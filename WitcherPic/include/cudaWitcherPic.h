#pragma once
#include "witcherPic_types.h"

#include <type_traits>
#include <cstdint>
#include <bits/shared_ptr.h>
#include <functional>


namespace witcher_pic {
	union Number {
		int8_t i8;
		int16_t i16;
		int32_t i32;
		int64_t i64;

		uint8_t ui8;
		uint16_t ui16;
		uint32_t ui32;
		uint64_t ui64;

		float f32;
		double f64;
		long double llf;
	};

	struct WitcherSize {
		unsigned width;
		unsigned height;
		const size_t size;

		WitcherSize(unsigned pw, unsigned ph);
	};
	
	template<bool _Const = true>
	using wsize_ptr = std::shared_ptr<std::conditional_t<_Const, const WitcherSize, WitcherSize>>;

	enum class WOperator {
		PLUS, MINUS, MUL, DEVIDE
	};

	enum class WCUDAResizeMode:uint8_t {
		NEAREST, BILINEAR, BICUBIC
	};

	extern "C" {
		auto deviceInit() -> void;
		auto hostDeviceInfo() -> void;
		auto hostMatFilter(uint8_t* source, float* model, const FilterInfo& info) -> uint8_t*;
		auto hostGrayCount(uint8_t* source, size_t size) -> size_t*;
		auto hostMapGrayImage(const uint8_t* source, const uint8_t* map_table, size_t size) -> void;
		auto hostTwoDimCrossCorre(const uint8_t* source, const float* model, unsigned s_w, unsigned s_h,
		                          unsigned m_w, unsigned m_h) -> uint8_t*;
		auto hostAddWeighted(size_t size, uint8_t* t, float w1, const uint8_t* s, float w2, uint8_t r) -> void;
		auto hostRGBAMatAssign(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a, const uint32_t* rgba, size_t size) -> void;
		auto hostInsertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t*;
		auto hostGetEdgeInfo(int* dirmat, const uint8_t* source, const float* model_x, const float* model_y,
		                     unsigned s_width, unsigned s_height, unsigned m_width, unsigned m_height,
		                     bool l2_gradient) -> uint8_t*;
		auto hostNonMaxSuppression(const uint8_t* source, const int* dir, unsigned width,
		                           unsigned height) -> uint8_t*;
		auto hostTwoThreshold(const uint8_t* source, unsigned width, unsigned height, uint8_t l_threshold,
		                      uint8_t h_threshold) -> uint8_t*;
		auto hostLineExtra(double** max_radius, double** max_thetas, size_t& line_size, const uint8_t* source,
		                   unsigned width, unsigned height, unsigned houghsize) -> void;
		auto hostDrawLine(uint8_t* source, unsigned width, unsigned height, double radius, double theta,
		                  uint8_t brightness, int thickness) -> void;
		auto hostRotate(const uint8_t* source, double theta, unsigned width, unsigned height,
		                unsigned& new_width, unsigned& new_height) -> uint8_t*;
		auto hostInterpo(WCUDAResizeMode mode, uint8_t* target, const uint8_t* source, wsize_ptr<> oldsize, wsize_ptr<> newsize) -> void;

		// haven't implemented yet
		auto hostCalc(WOperator op, Number* data1, Number* data2, ptrdiff_t data_size, size_t type_size,
		              bool is_floating, bool is_unsigned) -> Number*;
	}
}
