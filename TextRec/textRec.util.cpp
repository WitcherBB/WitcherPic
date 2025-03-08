module;
module textRec;
import <Eigen/Dense>;

#include "cudaTextRec.cuh"

using namespace Eigen;

namespace witcher_pic {
	template <typename T>
	auto copyMatData(const GenMatrix<T>& source) -> T* {
		auto size = source.rows() * source.cols();
		T* result = new T[size];
		memcpy(result, source.data(), size * sizeof(T));
		return result;
	}

	auto init() -> void {
		deviceInit();
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

		memcpy(img.r_matrix_.data(),
		       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
			       edgeinfo->r_dir->data(), img.r_matrix_.data(), x_model.data(), y_model.data(), width,
			       height, m_width, m_height, l2_gradient
		       )).get(), size);
		if (img.bpp_ != 8) {
			memcpy(img.g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       edgeinfo->g_dir->data(), img.g_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
			memcpy(img.b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       edgeinfo->b_dir->data(), img.b_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
		}
	}

	auto nonMaxSuppression(const EdgeInfo* e_info) -> void {
		Image* img = e_info->edge;
		memcpy(img->r_matrix_.data(),
		       std::shared_ptr<uint8_t[]>(
			       hostNonMaxSuppression(img->r_matrix_.data(), e_info->r_dir->data(), img->width(), img->height())
		       ).get(), img->size());
		if (img->bpp_ != 8) {
			memcpy(img->g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(
				       hostNonMaxSuppression(img->g_matrix_.data(), e_info->g_dir->data(), img->width(), img->height())
			       ).get(), img->size());
			memcpy(img->b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(
				       hostNonMaxSuppression(img->b_matrix_.data(), e_info->b_dir->data(), img->width(), img->height())
			       ).get(), img->size());
		}
	}

	auto twoThreshold(const EdgeInfo* e_info, uint8_t l_threshold, uint8_t h_threshold) -> void {
		Image* img = e_info->edge;
		auto r_data = img->r_matrix_.data();
		auto g_data = img->g_matrix_.data();
		auto b_data = img->b_matrix_.data();
		memcpy(r_data, std::shared_ptr<uint8_t[]>(
			       hostTwoThreshold(r_data, img->width(), img->height(), l_threshold, h_threshold)
		       ).get(), img->size());
		if (img->bpp_ != 8) {
			memcpy(g_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(g_data, img->width(), img->height(), l_threshold, h_threshold)
			       ).get(), img->size());
			memcpy(b_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(b_data, img->width(), img->height(), l_threshold, h_threshold)
			       ).get(), img->size());
		}
	}

	auto lineExtra(const ImgMat& source, unsigned houghsize) -> HoughInfo* {
		const auto width = source.cols();
		const auto height = source.rows();
		HoughInfo* hough = new HoughInfo{nullptr, nullptr, 0};

		hostLineExtra(&hough->max_redius, &hough->max_thetas, hough->size, source.data(), width, height,
		              houghsize);
		return hough;
	}

	auto drawLine(Image& img, double radius, double theta, uint32_t rgb, int thickness) -> void {
		uint8_t red = (uint8_t)(rgb >> 16 & 0xFF),
		        green = (uint8_t)(rgb >> 8 & 0xFF),
		        blue = (uint8_t)(rgb & 0xFF);
		unsigned width = img.width();
		unsigned height = img.height();

		ImageProcessor(img).toRGB();
		hostDrawLine(img.r_matrix_.data(), width, height, radius, theta, red, thickness);
		hostDrawLine(img.g_matrix_.data(), width, height, radius, theta, green, thickness);
		hostDrawLine(img.b_matrix_.data(), width, height, radius, theta, blue, thickness);
	}

	auto rotate(Image& img, double theta, bool clockwise) -> void {
		const auto width = img.width();
		const auto height = img.height();
		unsigned new_width(0), new_height(0);
		theta = clockwise ? theta : -theta;
		auto r_data = hostRotate(img.r_matrix_.data(), theta, width, height, new_width, new_height);
		unsigned new_size = new_width * new_height;
		Image new_img(new_width, new_height, img.bpp_);
		memcpy(new_img.r_matrix_.data(), r_data, new_size);
		delete[] r_data;
		if (img.bpp_) {
			memcpy(new_img.g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostRotate(img.g_matrix_.data(), theta, width, height, new_width,
			                                             new_height)).get(), new_size);
			memcpy(new_img.b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostRotate(img.b_matrix_.data(), theta, width, height, new_width,
			                                             new_height)).get(), new_size);
			if (img.bpp_ == 32) {
				memcpy(new_img.a_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostRotate(img.a_matrix_.data(), theta, width, height, new_width,
			                                             new_height)).get(), new_size);
			}
		}
		img = new_img;
	}

	auto gpuDeviceInfo() -> void {
		hostDeviceInfo();
	}
}
