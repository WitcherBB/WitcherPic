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

		uint8_t* result_arr = hostTwoDimCrossCorre(source.data(), model.data(), width, height, model.cols(), model.rows());

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

	auto getEdgeInfo(Image* img, bool l2_gradient) -> EdgeInfo* {
		const auto width = img->width();
		const auto height = img->height();
		const size_t size = width * height;

		EdgeInfo* e_info = new EdgeInfo{
			img,
			new EdgeInfo::EdgeDirMat(width, height),
			new EdgeInfo::EdgeDirMat(width, height),
			new EdgeInfo::EdgeDirMat(width, height)
		};

		const ModelMat& x_model = ModelMap::INSTANCE(SOBEL, 1);
		const ModelMat& y_model = ModelMap::INSTANCE(SOBEL, 2);
		unsigned m_width = x_model.cols();
		unsigned m_height = x_model.rows();

		memcpy(e_info->edge->r_matrix_.data(),
		       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
			       e_info->r_dir->data(), img->r_matrix_.data(), x_model.data(), y_model.data(), width,
			       height, m_width, m_height, l2_gradient
		       )).get(), size);
		if (img->bpp_ != 8) {
			memcpy(e_info->edge->g_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       e_info->g_dir->data(), img->g_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
			memcpy(e_info->edge->b_matrix_.data(),
			       std::shared_ptr<uint8_t[]>(hostGetEdgeInfo(
				       e_info->b_dir->data(), img->b_matrix_.data(), x_model.data(), y_model.data(), width,
				       height, m_width, m_height, l2_gradient
			       )).get(), size);
		}

		return e_info;
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

	auto gpuDeviceInfo() -> void {
		hostDeviceInfo();
	}
}
