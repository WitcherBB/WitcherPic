#include "witcherPic_util.h"
#include "cudaWitcherPic.h"

#include <ctime>
#include <stdexcept>
#include <numbers>
#include <Eigen/Dense>
#include <fmt/color.h>
#include <bits/shared_ptr.h>
#include <FreeImage.h>
#include "witcherPic.h"

using namespace Eigen;

#define SHARPEN_X 1
#define SHARPEN_Y 2

extern "C" {
	auto witcherpic_init() -> void {
		FreeImage_Initialise();
		witcher_pic::deviceInit();
	}

	auto witcherpic_deinit() -> void {
		FreeImage_DeInitialise();
	}
}

namespace witcher_pic {
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

	ModelMap::ModelMap(std::initializer_list<base_map::value_type> i_list): map_(i_list) {
		auto getter = [this](CppSharpenModel model, int index) -> ModelMat& {
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

	auto ModelMap::SModelPairHash::operator()(const std::pair<CppSharpenModel, int>& pair) const noexcept -> size_t {
		return static_cast<size_t>(pair.first) | static_cast<size_t>(pair.second) << 8;
	}

	auto ModelMap::operator()(CppSharpenModel model, int index) const -> const ModelMat& {
		return *map_.at(std::pair(model, index));
	}
}

namespace witcher_pic {
	auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy, FilterType type) -> ImageProcessor& {
		auto img_impl = processor.impl();
		img_impl->r_matrix_ = imgFilter(img_impl->r_matrix_, model, rcx, rcy, type);
		if (img_impl->bpp_ != 8) {
			img_impl->g_matrix_ = imgFilter(img_impl->g_matrix_, model, rcx, rcy, type);
			img_impl->b_matrix_ = imgFilter(img_impl->b_matrix_, model, rcx, rcy, type);
		}
		return processor;
	}

	EdgeInfo::~EdgeInfo() {
		delete r_dir;
		delete g_dir;
		delete b_dir;
	}

	Image::Image(unsigned width, unsigned height, int bpp)
		: p_impl_(new ImageImpl(width, height, bpp)) {
	}

    Image::Image(rgba *colors, unsigned width, unsigned height, int bpp)
		: p_impl_(new ImageImpl(colors, width, height, bpp)) {
    }

    Image::Image(const Image &mat)
        : p_impl_(new ImageImpl(*mat.p_impl_)) {
    }

    Image::~Image() {
		delete p_impl_;
	}

	auto Image::putPixel(unsigned x, unsigned y, rgba color) -> void {
		p_impl_->putPixel(x, y, color);
	}

	auto Image::putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void {
		p_impl_->putPixel(x, y, r, g, b, a);
	}

	auto Image::data() const -> uint8_t* {
		return p_impl_->data();
	}

    auto Image::normalData() const -> uint8_t*
    {
        return p_impl_->normalData();
    }

    auto Image::copy() const -> Image *
    {
        return new Image(*this);
    }

    auto Image::width() const -> unsigned {
		return p_impl_->width();
	}

	auto Image::height() const -> unsigned {
		return p_impl_->height();
	}

	auto Image::size() const -> size_t {
		return p_impl_->size();
	}

	auto Image::bpp() const -> int {
		return p_impl_->bpp_;
	}

	auto Image::impl() const -> ImageImpl* {
		return p_impl_;
	}

	auto Image::operator=(const Image& other) -> Image& {
		*p_impl_ = *other.p_impl_;
		return *this;
	}

	auto Image::operator()(unsigned x, unsigned y) const -> rgba {
		return (*p_impl_)(x, y);
	}

	template <typename T>
	auto copyMatData(const GenMatrix<T>& source) -> T* {
		auto size = source.rows() * source.cols();
		T* result = new T[size];
		memcpy(result, source.data(), size * sizeof(T));
		return result;
	}

	auto imgFilter(const ImgMat& source, const ModelMat& model, int rcx,
	               int rcy, FilterType type) -> ImgMat {
		unsigned width = source.cols();
		unsigned height = source.rows();

		uint8_t* source_arr = copyMatData(source);
		float* model_arr = copyMatData(model);
		uint8_t* result_arr = hostMatFilter(source_arr, model_arr,
		                                    FilterInfo(source.cols(), source.rows(), rcx, rcy, model.cols(),
		                                               model.rows(), type));

		ImgMat result;
		result.resize(height, width);
		memcpy(result.data(), result_arr, height * width);

		delete[] source_arr;
		delete[] model_arr;
		delete[] result_arr;

		return result;
	}

	auto grayCountTable(const ImgMat& source) -> size_t* {
		unsigned width = source.cols();
		unsigned height = source.rows();
		uint8_t* source_arr = copyMatData(source);

		size_t* gray_count = hostGrayCount(source_arr, width * height);

		delete[] source_arr;
		return gray_count;
	}

	auto mapGrayImage(const ImgMat& source, uint8_t* map_table,
	                  size_t size) -> void {
		hostMapGrayImage(source.data(), map_table, size);
	}

	auto imageSharpen(const ImgMat& source, const ModelMat& model) -> ImgMat {
		unsigned width = source.cols();
		unsigned height = source.rows();

		uint8_t* result_arr = hostTwoDimCrossCorre(source.data(), model.data(), width, height, model.cols(),
		                                           model.rows());

		ImgMat result;
		result.resize(height, width);
		memcpy(result.data(), result_arr, width * height);

		return result;
	}

	auto imageAddWeighted(ImgMat& target, float weight1, const ImgMat& other, float weight2,
	                      uint8_t r) -> void {
		size_t size = target.cols() * other.rows();
		hostAddWeighted(size, target.data(), weight1, other.data(), weight2, r);
	}

	auto gaussianKernel(ModelMat& model, float sigma) -> void {
		auto width = model.cols();
		auto size = width * width;
		auto middle = (width - 1) / 2;

		const double PI = std::numbers::pi;
		const double E = std::numbers::e;

		float total = 0;

		auto gaussian = [&](Index x, Index y) -> float {
			return 1.0 / (2 * PI * sigma * sigma) * pow(E, -1.0 * (x * x + y * y) / (2 * sigma * sigma));
		};

		for (int i = 0; i < size; i++) {
			auto x_i = i % width;
			auto y_i = i / width;

			model(i) = gaussian(x_i - middle, y_i - middle);
			total += model(i);
		}
		model /= total;
	}

	auto insertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t* {
		return hostInsertData(data, datasize, count);
	}

	auto getEdgeInfo(EdgeInfo* edgeinfo, Image& img, bool l2_gradient) -> void {
		const auto width = img.width();
		const auto height = img.height();
		const size_t size = width * height;

		edgeinfo->edge = &img;
		edgeinfo->r_dir = new EdgeInfo::EdgeDirMat(width, height);
		edgeinfo->g_dir = new EdgeInfo::EdgeDirMat(width, height);
		edgeinfo->b_dir = new EdgeInfo::EdgeDirMat(width, height);

		const ModelMat& x_model = ModelMap::INSTANCE(SOBEL, 1);
		const ModelMat& y_model = ModelMap::INSTANCE(SOBEL, 2);
		unsigned m_width = x_model.cols();
		unsigned m_height = x_model.rows();

		auto impl = img.impl();
		memcpy(impl->r_matrix_.data(),
		       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
			       edgeinfo->r_dir->data(), impl->r_matrix_.data(), x_model.data(), y_model.data(), width,
			       height, m_width, m_height, l2_gradient
		       )).get(), size);
		if (impl->bpp_ != 8) {
			memcpy(impl->g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       edgeinfo->g_dir->data(), impl->g_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
			memcpy(impl->b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       edgeinfo->b_dir->data(), impl->b_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
		}
	}

	auto nonMaxSuppression(const EdgeInfo* e_info) -> void {
		auto impl = e_info->edge->impl();
		memcpy(impl->r_matrix_.data(),
		       std::shared_ptr<uint8_t[]>(
			       hostNonMaxSuppression(impl->r_matrix_.data(), e_info->r_dir->data(), impl->width(), impl->height())
		       ).get(), impl->size());
		if (impl->bpp_ != 8) {
			memcpy(impl->g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(
				       hostNonMaxSuppression(impl->g_matrix_.data(), e_info->g_dir->data(), impl->width(),
				                             impl->height())
			       ).get(), impl->size());
			memcpy(impl->b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(
				       hostNonMaxSuppression(impl->b_matrix_.data(), e_info->b_dir->data(), impl->width(),
				                             impl->height())
			       ).get(), impl->size());
		}
	}

	auto twoThreshold(const EdgeInfo* e_info, const uint8_t* l_threshold, const uint8_t* h_threshold) -> void {
		auto impl = e_info->edge->impl();
		auto r_data = impl->r_matrix_.data();
		auto g_data = impl->g_matrix_.data();
		auto b_data = impl->b_matrix_.data();
		memcpy(r_data, std::shared_ptr<uint8_t[]>(
			       hostTwoThreshold(r_data, impl->width(), impl->height(), l_threshold[0], h_threshold[0])
		       ).get(), impl->size());
		if (impl->bpp_ != 8) {
			memcpy(g_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(g_data, impl->width(), impl->height(), l_threshold[1], h_threshold[1])
			       ).get(), impl->size());
			memcpy(b_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(b_data, impl->width(), impl->height(), l_threshold[2], h_threshold[2])
			       ).get(), impl->size());
		}
	}

	auto lineExtra(const ImgMat& source, double rho, double theta, size_t threshold) -> HoughInfo* {
		const auto width = source.cols();
		const auto height = source.rows();
		HoughInfo* hough = hostLineExtra(source.data(), width, height, rho, theta, threshold);
		return hough;
	}

	auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void {
		uint8_t red = (uint8_t)(rgb >> 16 & 0xFF),
		        green = (uint8_t)(rgb >> 8 & 0xFF),
		        blue = (uint8_t)(rgb & 0xFF);
		unsigned width = img.width();
		unsigned height = img.height();
		auto impl = img.impl();

		ImageProcessor(img).toRGB();
		hostDrawLine(impl->r_matrix_.data(), width, height, radius, theta, red, thickness);
		hostDrawLine(impl->g_matrix_.data(), width, height, radius, theta, green, thickness);
		hostDrawLine(impl->b_matrix_.data(), width, height, radius, theta, blue, thickness);
	}

    auto drawLines(Image& img, const HoughInfo& houghinfo, uint32_t rgb, int thickness) -> void {
		uint8_t red = (uint8_t)(rgb >> 16 & 0xFF),
		        green = (uint8_t)(rgb >> 8 & 0xFF),
		        blue = (uint8_t)(rgb & 0xFF);
		unsigned width = img.width();
		unsigned height = img.height();
		auto impl = img.impl();

		ImageProcessor(img).toRGB();
		//DOWN HoughInfo
		for (size_t i = 0; i < houghinfo.size; i++) {
			double radius = houghinfo.max_radius[i];
			double theta = houghinfo.max_thetas[i];
			hostDrawLine(impl->r_matrix_.data(), width, height, radius, theta, red, thickness);
			hostDrawLine(impl->g_matrix_.data(), width, height, radius, theta, green, thickness);
			hostDrawLine(impl->b_matrix_.data(), width, height, radius, theta, blue, thickness);
		}
    }

    auto rotate(Image& img, double theta, bool clockwise) -> void {
		auto impl = img.impl();
		const auto width = img.width();
		const auto height = img.height();
		unsigned new_width(0), new_height(0);
		theta = clockwise ? theta : -theta;
		auto r_data = hostRotate(impl->r_matrix_.data(), theta, width, height, new_width, new_height);
		unsigned new_size = new_width * new_height;
		Image new_img(new_width, new_height, impl->bpp_);
		auto new_impl = new_img.impl();
		memcpy(new_impl->r_matrix_.data(), r_data, new_size);
		delete[] r_data;
		if (impl->bpp_) {
			memcpy(new_impl->g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostRotate(impl->g_matrix_.data(), theta, width, height, new_width,
			                                             new_height)).get(), new_size);
			memcpy(new_impl->b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostRotate(impl->b_matrix_.data(), theta, width, height, new_width,
			                                             new_height)).get(), new_size);
			if (impl->bpp_ == 32) {
				memcpy(new_impl->a_matrix_.data(),
				       std::shared_ptr<uint8_t[]>(hostRotate(impl->a_matrix_.data(), theta, width, height, new_width,
				                                             new_height)).get(), new_size);
			}
		}
		img = new_img;
	}

	auto gpuDeviceInfo() -> void {
		hostDeviceInfo();
	}

	
}
