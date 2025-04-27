#pragma once
#include "witcherPic.h"
#include "witcherPic_types.h"
#include <Eigen/Dense>

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
	class Image;
	class ImageProcessor;
	class ImageImpl;

	template <typename T>
	using GenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using ImgMat = GenMatrix<uint8_t>;
	using ModelMat = GenMatrix<float>;
	using HoughZoom = GenMatrix<size_t>;

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

	class ModelMap {
		class SModelPairHash {
		public:
			auto operator()(const std::pair<CppSharpenModel, int>& pair) const noexcept -> size_t;
		};

	public:
		using hasher = SModelPairHash;
		using base_map = std::unordered_map<std::pair<CppSharpenModel, int>, ModelMat*, hasher>;

		ModelMap() = delete;
		ModelMap(const ModelMap&) = delete;
		ModelMap(ModelMap&&) = delete;
		auto operator()(CppSharpenModel model, int index) const -> const ModelMat&;

		static const ModelMap INSTANCE;

	private:
		ModelMap(std::initializer_list<base_map::value_type> i_list);
		const base_map map_;
	};

	class ImageImpl {
		friend class Image;
		friend class ImageProcessor;
		friend auto getEdgeInfo(EdgeInfo* edgeinfo, Image& img, bool l2_gradient) -> void;
		friend auto nonMaxSuppression(const EdgeInfo* e_info) -> void;
		friend auto twoThreshold(const EdgeInfo* e_info, const uint8_t* l_threshold,
								const uint8_t* h_threshold) -> void;
		friend auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void;
		friend auto rotate(Image& img, double theta, bool clockwise) -> void;
		friend auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy, FilterType type) -> ImageProcessor&;

		enum SharpenMode {
			NORMAL, MIXs
		};

	public:
		ImageImpl(unsigned width, unsigned height, int bpp);
		ImageImpl(rgba* colors, unsigned width, unsigned height, int bpp);
		ImageImpl(const ImageImpl& mat);
		auto resize(unsigned width, unsigned height) -> void;
		auto resizeLike(const ImageImpl& other) -> void;
		auto putPixel(unsigned x, unsigned y, rgba color) -> void;
		auto putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void;
		auto data() const -> uint8_t*;
		auto normalData() const -> uint8_t*;

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

	auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy, FilterType type) -> ImageProcessor&;

#ifdef _DEBUG
	{
#endif
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
	auto twoThreshold(const EdgeInfo* e_info, const uint8_t* l_threshold, const uint8_t* h_threshold) -> void;
	auto lineExtra(const ImgMat& source, unsigned houghsize) -> HoughInfo*;
	auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void;
	auto rotate(Image& img, double theta, bool clockwise = false) -> void;

#ifdef _DEBUG
	}
#endif
}
