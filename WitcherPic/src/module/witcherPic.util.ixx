module;
#include "witcherPic_types.h"

#include <exception>
#include <numbers>
#include <Eigen/Dense>
export module witcherPic:util;

#define SHARPEN_X 1
#define SHARPEN_Y 2

using namespace Eigen;

template <typename F>
class Finally {
	F func_;

public:
	Finally(F&& func): func_(std::forward<F>(func)) {
	}

	~Finally() {
		func_();
	}
};

namespace witcher_pic {
	export class Image;
	export class ImageProcessor;
	export struct EdgeInfo;
	export struct HoughInfo;
	class ImageImpl;

	export template <typename T>
	using GenMatrix = Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>;
	export using ImgMat = GenMatrix<uint8_t>;
	export using ModelMat = GenMatrix<float>;
	using HoughZoom = GenMatrix<size_t>;
	export using rgba = uint32_t;

	export enum SharpenModel:uint8_t {
		LAPLACIAN,
		SOBEL,
		ROBERTS,
		PREWITT,
		LOG
	};

	auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy, FilterType type) -> ImageProcessor&;

	export class ModelMap {
		class SModelPairHash {
		public:
			auto operator()(const std::pair<SharpenModel, int>& pair) const noexcept -> size_t {
				return static_cast<size_t>(pair.first) | static_cast<size_t>(pair.second) << 8;
			}
		};

	public:
		using hasher = SModelPairHash;
		using base_map = std::unordered_map<std::pair<SharpenModel, int>, ModelMat*, hasher>;

		ModelMap() = delete;
		ModelMap(const ModelMap&) = delete;
		ModelMap(ModelMap&&) = delete;


		auto operator()(SharpenModel model, int index) const -> const ModelMat& {
			return *map_.at(std::pair(model, index));
		}

		static const ModelMap INSTANCE;

	private:
		ModelMap(std::initializer_list<base_map::value_type> i_list): map_(i_list) {
			auto getter = [this](SharpenModel model, int index) -> ModelMat& {
				return *map_.at(std::pair(model, index));
			};

			getter(LAPLACIAN, 0) <<
				0, -1, 0,
				-1, 4, -1,
				0, -1, 0;
			getter(LAPLACIAN, 1) <<
				-1, -1, -1,
				-1, 8, -1,
				-1, -1, -1;
			getter(SOBEL, 1) <<
				-1, 0, 1,
				-2, 0, 2,
				-1, 0, 1;
			getter(SOBEL, 2) <<
				-1, -2, -1,
				0, 0, 0,
				1, 2, 1;
			getter(ROBERTS, 1) <<
				-1, 0,
				0, 1;
			getter(ROBERTS, 2) <<
				0, -1,
				1, 0;
			getter(PREWITT, 1) <<
				-1, 0, 1,
				-1, 0, 1,
				-1, 0, 1;
			getter(PREWITT, 2) <<
				-1, -1, -1,
				0, 0, 0,
				1, 1, 1;
			getter(LOG, 0) <<
				-2, -4, -4, -4, -2,
				-4, 0, 8, 0, -4,
				-4, 8, 24, 8, -4,
				-4, 0, 8, 0, -4,
				-2, -4, -4, -4, -2;
			getter(LOG, 1) <<
				0, 1, 1, 2, 2, 2, 1, 1, 0,
				1, 2, 4, 5, 5, 5, 4, 2, 1,
				1, 4, 5, 3, 0, 3, 5, 4, 1,
				2, 5, 3, -12, -24, -12, 3, 5, 2,
				2, 5, 0, -24, -40, -24, 0, 5, 2,
				2, 5, 3, -12, -24, -12, 3, 5, 2,
				1, 4, 5, 3, 0, 3, 5, 4, 1,
				1, 2, 4, 5, 5, 5, 4, 2, 1,
				0, 1, 1, 2, 2, 2, 1, 1, 0;
		}

		const base_map map_;
	};

	const ModelMap ModelMap::INSTANCE = ModelMap({
		std::pair(std::pair(LAPLACIAN, 0), new ModelMat(3, 3)),
		std::pair(std::pair(LAPLACIAN, 1), new ModelMat(3, 3)),
		std::pair(std::pair(SOBEL, 0), new ModelMat(3, 3)),
		std::pair(std::pair(SOBEL, 1), new ModelMat(3, 3)),
		std::pair(std::pair(SOBEL, 2), new ModelMat(3, 3)),
		std::pair(std::pair(ROBERTS, 0), new ModelMat(2, 2)),
		std::pair(std::pair(ROBERTS, 1), new ModelMat(2, 2)),
		std::pair(std::pair(ROBERTS, 2), new ModelMat(2, 2)),
		std::pair(std::pair(PREWITT, 0), new ModelMat(3, 3)),
		std::pair(std::pair(PREWITT, 1), new ModelMat(3, 3)),
		std::pair(std::pair(PREWITT, 2), new ModelMat(3, 3)),
		std::pair(std::pair(LOG, 0), new ModelMat(5, 5)),
		std::pair(std::pair(LOG, 1), new ModelMat(9, 9)),
	});

#ifdef _DEBUG
	export auto imgFilter(const ImgMat& source, const ModelMat& model, int rcx,
	                      int rcy, FilterType type = CONV) -> ImgMat;
	export auto grayCountTable(const ImgMat& source) -> size_t*;
	export auto mapGrayImage(const ImgMat& source, uint8_t* map_table,
	                         size_t size) -> void;
	export auto imageSharpen(const ImgMat& source, const ModelMat& model) -> ImgMat;
	export auto imageAddWeighted(ImgMat& target, float weight1, const ImgMat& other, float weight2,
	                             uint8_t r) -> void;
	export auto gaussianKernel(ModelMat& model, float sigma) -> void;
	export auto insertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t*;
	export auto getEdgeInfo(EdgeInfo* edgeinfo, Image& img, bool l2_gradient) -> void;
	export auto nonMaxSuppression(const EdgeInfo* e_info) -> void;
	export auto twoThreshold(const EdgeInfo* e_info, uint8_t l_threshold, uint8_t h_threshold) -> void;
	export auto lineExtra(const ImgMat& source, unsigned houghsize) -> HoughInfo*;
	export auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void;
	export auto rotate(Image& img, double theta, bool clockwise = false) -> void;

	export auto gpuDeviceInfo() -> void;
#else
	auto imgFilter(const ImgMat& source, const ModelMat& model, int rcx,
	               int rcy, FilterType type = CONV) -> ImgMat;
	auto grayCountTable(const ImgMat& source) -> size_t*;
	auto mapGrayImage(const ImgMat& source, uint8_t* map_table,
	                  size_t size) -> void;
	auto imageSharpen(const ImgMat& source, const ModelMat& model) -> ImgMat;
	auto imageAddWeighted(ImgMat& target, float weight1, const ImgMat& other, float weight2,
	                      uint8_t r) -> void;
	auto gaussianKernel(ModelMat& model, float sigma) -> void;
	auto insertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t*;
	auto getEdgeInfo(EdgeInfo* edgeinfo, Image& img, bool l2_gradient) -> void;
	auto nonMaxSuppression(const EdgeInfo* e_info) -> void;
	auto twoThreshold(const EdgeInfo* e_info, uint8_t l_threshold, uint8_t h_threshold) -> void;
	auto lineExtra(const ImgMat& source, unsigned houghsize) -> HoughInfo*;
	auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void;
	auto rotate(Image& img, double theta, bool clockwise = false) -> void;
#endif
}

namespace witcher_pic {
	struct HoughInfo {
		double* max_redius;
		double* max_thetas;
		size_t size;

		~HoughInfo();
	};

	struct EdgeInfo {
		using EdgeDirMat = GenMatrix<int>;

		Image* edge;
		EdgeDirMat* r_dir;
		EdgeDirMat* g_dir;
		EdgeDirMat* b_dir;

		~EdgeInfo();
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

	class ImageImpl {
		friend class Image;
		friend class ImageProcessor;
		friend auto getEdgeInfo(EdgeInfo* edgeinfo, Image& img, bool l2_gradient) -> void;
		friend auto nonMaxSuppression(const EdgeInfo* e_info) -> void;
		friend auto twoThreshold(const EdgeInfo* e_info, uint8_t l_threshold, uint8_t h_threshold) -> void;
		friend auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void;
		friend auto rotate(Image& img, double theta, bool clockwise) -> void;
		friend auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy,
		                   FilterType type) -> ImageProcessor&;

		enum SharpenMode {
			NORMAL, MIXs
		};

	public:
		ImageImpl(unsigned width, unsigned height, int bpp);
		ImageImpl(const ImageImpl& mat);
		auto resize(unsigned width, unsigned height) -> void;
		auto resizeLike(const ImageImpl& other) -> void;
		auto putPixel(unsigned x, unsigned y, rgba color) -> void;
		auto putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void;
		auto data() const -> uint8_t*;

		auto width() const -> unsigned;
		auto height() const -> unsigned;
		auto size() const -> size_t;
		auto bpp() const -> int;

		auto operator=(const ImageImpl& other) -> ImageImpl&;
		auto operator()(unsigned x, unsigned y) const -> rgba;

	protected:
		ImgMat r_matrix_;
		ImgMat g_matrix_;
		ImgMat b_matrix_;
		ImgMat a_matrix_;
		int bpp_;
	};

	class ImageProcessor {
		friend auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy,
		                   FilterType type) -> ImageProcessor&;

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
