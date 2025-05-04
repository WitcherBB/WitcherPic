#pragma once
#include <stddef.h>
#include <stdint.h>

namespace witcher_pic {
	using rgba = uint32_t;
	
	class Image;
	class ImageImpl;

	struct EdgeInfo;
}


namespace witcher_pic {
	enum CppSharpenModel:uint8_t {
		LAPLACIAN,
		SOBEL,
		ROBERTS,
		PREWITT,
		LOG
	};

	enum CppResizeMode:uint8_t {
		NEAREST,
		BILINEAR,
		BICUBIC
	};

	class Image {
	public:
		Image(unsigned width, unsigned height, int bpp);
		Image(rgba* colors, unsigned width, unsigned height, int bpp);
		Image(const Image& mat);
		~Image();
		auto putPixel(unsigned x, unsigned y, rgba color) -> void;
		auto putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void;
		auto data() const -> uint8_t*;
		auto normalData() const-> uint8_t*;
		auto copy() const -> Image*;

		auto width() const -> unsigned;
		auto height() const -> unsigned;
		auto size() const -> size_t;
		auto bpp() const -> int;
		auto impl() const -> ImageImpl*;

		auto operator=(const Image& other) -> Image&;
		auto operator()(unsigned x, unsigned y) const -> rgba;

	protected:
		ImageImpl* p_impl_;
	};

	class ImageProcessor {
		enum SharpenMode {
			NORMAL, MIX
		};

	public:
		ImageProcessor(Image* img);
		ImageProcessor(Image& img);

		auto averFilter(unsigned size) -> ImageProcessor&;
		auto medianFilter(unsigned size) -> ImageProcessor&;
		auto gaussianFilter(unsigned size, float sigma) -> ImageProcessor&;
		auto toRGBA() -> ImageProcessor&;
		auto toRGB() -> ImageProcessor&;
		auto toGray() -> ImageProcessor&;
		auto toBinary(uint8_t m) -> ImageProcessor&;
		auto toOtsuBinary() -> ImageProcessor&;
		auto grayEnhance(float min_rate, float max_rate) -> ImageProcessor&;
		auto edgeExtra(CppSharpenModel model, int index) -> ImageProcessor&;
		auto sharpen(CppSharpenModel model, float strength, int index) -> ImageProcessor&;
		auto canny(unsigned kernelsize, bool l2_gradient) -> ImageProcessor&;
		auto canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize, bool l2_gradient) -> ImageProcessor&;
		// Private
		auto canny(EdgeInfo& edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
		           bool l2_gradient) -> ImageProcessor&;
		// Private
		auto canny(EdgeInfo* edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
		           bool l2_gradient) -> ImageProcessor&;
		auto tiltCorrection(size_t zoomsize, bool copy = false, unsigned kermelsize = 3,
		                    bool l2_gradient = true, bool pdebug = false) -> ImageProcessor&;
		auto addWeighted(const Image& other, float w1, float w2,
		                 uint8_t r = 0) -> ImageProcessor&;
		auto addWeighted(const Image* other, float w1, float w2,
		                 uint8_t r = 0) -> ImageProcessor&;
		auto resize(long newwidth = -1, long newheight = -1, CppResizeMode mode = CppResizeMode::NEAREST) -> ImageProcessor&;
		auto get() const -> Image&;
		auto impl() const -> ImageImpl*;

	private:
		[[nodiscard]] static auto checkModelIndex(CppSharpenModel model, int index) -> SharpenMode;
		static auto calcThresholdWithOtsu(const Image* img) -> uint8_t*;

		Image* img_;
	};

	auto gpuDeviceInfo() -> void;
}

extern "C" {
	auto witcherpic_init() -> void;
	auto witcherpic_deinit() -> void;
	auto witcherpic_recognizeText(const char* pic_name) -> void;
	auto witcherpic_mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r = 0) -> void;
	auto witcherpic_loadImage(const char* pic_name) -> witcher_pic::Image*;
	auto witcherpic_refSaveImage(const char* filename, const witcher_pic::Image& image) -> void;
	auto witcherpic_ptrSaveImage(const char* filename, const witcher_pic::Image* image) -> void;
}
