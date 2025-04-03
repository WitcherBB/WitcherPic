#pragma once
#include <memory>

namespace witcher_pic {
	class ImageImpl;
	class Image;
	using uint8_t = unsigned char;
	using uint32_t = unsigned int;
	using rgba = uint32_t;
	struct HoughInfo;
	struct EdgeInfo;

	enum SharpenModel:uint8_t {
		LAPLACIAN,
		SOBEL,
		ROBERTS,
		PREWITT,
		LOG
	};

	class Image {
	public:
		Image(unsigned width, unsigned height, int bpp);
		Image(const Image& mat);
		~Image();
		auto resize(unsigned width, unsigned height) -> void;
		auto resizeLike(const Image& other) -> void;
		auto resizeLike(const Image* other) -> void;
		auto putPixel(unsigned x, unsigned y, rgba color) -> void;
		auto putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void;
		auto data() const -> uint8_t*;
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
		auto edgeExtra(SharpenModel model, int index) -> ImageProcessor&;
		auto sharpen(SharpenModel model, float strength, int index) -> ImageProcessor&;
		auto canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize = 3,
		           bool l2_gradient = false) -> ImageProcessor&;
		// Private
		auto edgeCanny(EdgeInfo& edgeinfo, uint8_t l_threshold, uint8_t h_threshold,
		           unsigned kernelsize = 3, bool l2_gradient = false) -> ImageProcessor&;
		// Private
		auto edgeCanny(EdgeInfo* edgeinfo, uint8_t l_threshold, uint8_t h_threshold,
		           unsigned kernelsize = 3, bool l2_gradient = false) -> ImageProcessor&;
		auto tiltCorrection(size_t zoomsize, bool copy = false) -> ImageProcessor&;
		auto addWeighted(const Image& other, float w1, float w2,
		                 uint8_t r = 0) -> ImageProcessor&;
		auto addWeighted(const Image* other, float w1, float w2,
		                 uint8_t r = 0) -> ImageProcessor&;
		auto get() const -> Image&;

	private:
		[[nodiscard]] static auto checkModelIndex(SharpenModel model, int index) -> SharpenMode;

		Image* img_;
		ImageImpl* impl_;
	};
}

extern "C" {
auto witcherpic_init() -> void;
auto witcherpic_recognizeText(const char* pic_name) -> void;
auto witcherpic_mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2,
                         uint8_t r = 0) -> void;
auto witcherpic_loadImage(const char* pic_name) -> witcher_pic::Image*;
auto witcherpic_refSaveImage(const char* filename, const witcher_pic::Image& image) -> void;
auto witcherpic_ptrSaveImage(const char* filename, const witcher_pic::Image* image) -> void;
}
