module;
#include "cudaWitcherPic.h"
#include "witcherPic_types.h"

#include <ctime>
#include <numbers>
#include <Eigen/Dense>
module witcherPic;

using namespace Eigen;

#define SHARPEN_X 1
#define SHARPEN_Y 2

extern "C" {
auto witcherpic_init() -> void {
	witcher_pic::deviceInit();
}
}

namespace witcher_pic {
	auto filter(ImageProcessor& processor, const ModelMat& model, int rcx, int rcy, FilterType type) -> ImageProcessor& {
		processor.impl_->r_matrix_ = imgFilter(processor.impl_->r_matrix_, model, rcx, rcy, type);
		if (processor.impl_->bpp_ != 8) {
			processor.impl_->g_matrix_ = imgFilter(processor.impl_->g_matrix_, model, rcx, rcy, type);
			processor.impl_->b_matrix_ = imgFilter(processor.impl_->b_matrix_, model, rcx, rcy, type);
		}
		return processor;
	}

	HoughInfo::~HoughInfo() {
		delete[] max_redius;
		delete[] max_thetas;
	}

	EdgeInfo::~EdgeInfo() {
		delete r_dir;
		delete g_dir;
		delete b_dir;
	}

	Image::Image(unsigned width, unsigned height, int bpp): p_impl_(new ImageImpl(width, height, bpp)) {
	}

	Image::Image(const Image& mat)
		: p_impl_(new ImageImpl(mat.impl()->width(), mat.impl()->height(), mat.impl()->bpp_)) {
	}

	Image::~Image() {
		delete p_impl_;
	}

	auto Image::resize(unsigned width, unsigned height) -> void {
		p_impl_->resize(width, height);
	}

	auto Image::resizeLike(const Image& other) -> void {
		p_impl_->resizeLike(*other.impl());
	}

	auto Image::resizeLike(const Image* other) -> void {
		p_impl_->resizeLike(*other->impl());
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

	auto Image::copy() const -> Image* {
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

	ImageImpl::ImageImpl(unsigned width, unsigned height, int bpp): bpp_(bpp) {
		r_matrix_.resize(height, width);
		r_matrix_.fill(0u);
		g_matrix_.resize(height, width);
		g_matrix_.fill(0u);
		b_matrix_.resize(height, width);
		b_matrix_.fill(0u);
		a_matrix_.resize(height, width);
		a_matrix_.fill(255u);
	}

	ImageImpl::ImageImpl(const ImageImpl& mat): r_matrix_(mat.r_matrix_), g_matrix_(mat.g_matrix_),
	                                            b_matrix_(mat.b_matrix_),
	                                            a_matrix_(mat.a_matrix_), bpp_(mat.bpp_) {
	}

	auto ImageImpl::resize(unsigned width, unsigned height) -> void {
		r_matrix_.conservativeResize(height, width);
		g_matrix_.conservativeResize(height, width);
		b_matrix_.conservativeResize(height, width);
		a_matrix_.conservativeResize(height, width);
	}

	auto ImageImpl::resizeLike(const ImageImpl& other) -> void {
		r_matrix_.conservativeResizeLike(other.r_matrix_);
		g_matrix_.conservativeResizeLike(other.g_matrix_);
		b_matrix_.conservativeResizeLike(other.b_matrix_);
		a_matrix_.conservativeResizeLike(other.a_matrix_);
	}

	auto ImageImpl::putPixel(unsigned x, unsigned y, rgba color) -> void {
		r_matrix_(y, x) = static_cast<uint8_t>((color >> 24) & 0xFF);
		g_matrix_(y, x) = static_cast<uint8_t>((color >> 16) & 0xFF);
		b_matrix_(y, x) = static_cast<uint8_t>((color >> 8) & 0xFF);
		a_matrix_(y, x) = static_cast<uint8_t>(color & 0xFF);
	}

	auto ImageImpl::putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void {
		r_matrix_(y, x) = r;
		g_matrix_(y, x) = g;
		b_matrix_(y, x) = b;
		a_matrix_(y, x) = a;
	}

	auto ImageImpl::data() const -> uint8_t* {
		switch (bpp_) {
		case 32:
			const uint8_t* bgra_data[4];
			bgra_data[0] = b_matrix_.data();
			bgra_data[1] = g_matrix_.data();
			bgra_data[2] = r_matrix_.data();
			bgra_data[3] = a_matrix_.data();
			return insertData(bgra_data, size(), 4);
		case 24:
			const uint8_t* bgr_data[3];
			bgr_data[0] = b_matrix_.data();
			bgr_data[1] = g_matrix_.data();
			bgr_data[2] = r_matrix_.data();
			return insertData(bgr_data, size(), 3);
		case 8:
			uint8_t* gray_data = new uint8_t[size()];
			memcpy(gray_data, r_matrix_.data(), size());
			return gray_data;
		}
		throw std::exception("bpp wrong!");
	}

	auto ImageImpl::width() const -> unsigned {
		return r_matrix_.cols();
	}

	auto ImageImpl::height() const -> unsigned {
		return r_matrix_.rows();
	}

	auto ImageImpl::size() const -> size_t {
		return width() * height();
	}

	auto ImageImpl::bpp() const -> int {
		return bpp_;
	}

	auto ImageImpl::operator=(const ImageImpl& other) -> ImageImpl& {
		r_matrix_ = other.r_matrix_;
		g_matrix_ = other.g_matrix_;
		b_matrix_ = other.b_matrix_;
		a_matrix_ = other.a_matrix_;
		bpp_ = other.bpp_;
		return *this;
	}

	auto ImageImpl::operator()(unsigned x, unsigned y) const -> rgba {
		return static_cast<rgba>(
			r_matrix_(y, x) << 24 |
			g_matrix_(y, x) << 16 |
			b_matrix_(y, x) << 8 |
			a_matrix_(y, x)
		);
	}

	ImageProcessor::ImageProcessor(Image* img): img_(img), impl_(img->impl()) {
	}

	ImageProcessor::ImageProcessor(Image& img): img_(&img), impl_(img.impl()) {
	}

	auto ImageProcessor::averFilter(unsigned size) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::exception("The Gaussian template size must be odd.");
		}
		ModelMat model(size, size);
		model.fill(1.0 / (size * size));
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto ImageProcessor::medianFilter(unsigned size) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::exception("midianFilter: template size must be odd.");
		}
		ModelMat model(size, size);
		model.fill(1);
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, MEDIAN);
	}

	auto ImageProcessor::gaussianFilter(unsigned size, float sigma) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::exception("gaussianFilter: template size must be odd.");
		}

		ModelMat model(size, size);
		gaussianKernel(model, sigma);
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto ImageProcessor::toRGBA() -> ImageProcessor& {
		if (impl_->bpp_ != 32) {
			impl_->a_matrix_.fill(255);
			if (impl_->bpp_ == 8) {
				impl_->g_matrix_ = impl_->r_matrix_;
				impl_->b_matrix_ = impl_->r_matrix_;
			}
		}
		impl_->bpp_ = 32;
		return *this;
	}

	auto ImageProcessor::toRGB() -> ImageProcessor& {
		if (impl_->bpp_ == 8) {
			impl_->g_matrix_ = impl_->r_matrix_;
			impl_->b_matrix_ = impl_->r_matrix_;
		}
		impl_->bpp_ = 24;
		return *this;
	}

	auto ImageProcessor::toGray() -> ImageProcessor& {
		impl_->bpp_ = 8;
		for (size_t i = 0; i < impl_->size(); i++) {
			uint8_t gray = static_cast<uint8_t>(
				static_cast<float>(impl_->r_matrix_(i)) * 0.299F +
				static_cast<float>(impl_->g_matrix_(i)) * 0.587F +
				static_cast<float>(impl_->b_matrix_(i)) * 0.114F
			);
			impl_->r_matrix_(i) = gray;
		}
		return *this;
	}

	auto ImageProcessor::toBinary(uint8_t m) -> ImageProcessor& {
		if (impl_->bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return *this;
		}

		for (auto y = 0u; y < impl_->height(); ++y) {
			for (auto x = 0u; x < impl_->width(); ++x) {
				impl_->r_matrix_(y, x) = impl_->r_matrix_(y, x) >= m ? 255 : 0;;
			}
		}
		return *this;
	}

	auto ImageProcessor::toOtsuBinary() -> ImageProcessor& {
		if (impl_->bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return *this;
		}

		size_t* gray_table = grayCountTable(impl_->r_matrix_);
		double variance[256] = {0};
		auto img_size = impl_->size();
		uint8_t L = 0;
		auto pi = new double[256];
		for (unsigned i = 0; i <= 255; i++) {
			if (gray_table[i]) {
				L = i;
			}
			pi[i] = static_cast<double>(gray_table[i]) / img_size;
		}

		for (unsigned i = 1; i < L; i++) {
			auto count1 = 0.0;
			auto count2 = 0.0;
			// test
			for (unsigned j = 0; j < i; j++) {
				count1 += gray_table[j];
			}
			// test
			for (unsigned j = i; j <= L; j++) {
				count2 += gray_table[j];
			}
			auto w1 = count1 / img_size;
			auto w2 = count2 / img_size;
			double u1 = 0;
			double u2 = 0;
			for (unsigned j = 0; j < i; j++) {
				u1 += j * pi[j];
			}
			for (unsigned j = i; j <= L; j++) {
				u1 += j * pi[j];
			}

			variance[i] = w1 * w2 * (u1 - u2) * (u1 - u2);
		}

		uint8_t m = 0;
		for (unsigned i = 0; i <= L; ++i) {
			if (variance[m] < variance[i]) {
				m = i;
			}
		}
		delete[] gray_table;
		return toBinary(m);
	}

	auto ImageProcessor::grayEnhance(float min_rate, float max_rate) -> ImageProcessor& {
		if (impl_->bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return *this;
		}
		if (min_rate > 100 || max_rate > 100 || min_rate + max_rate > 100 || min_rate < 0 || max_rate < 0) {
			fprintf(stderr, "rate wrong!\n");
			return *this;
		}

		size_t min_thre = static_cast<size_t>(impl_->size() / 100.0 * min_rate);
		size_t max_thre = static_cast<size_t>(impl_->size() / 100.0 * max_rate);
		uint8_t min_gray = 0;
		uint8_t max_gray = 0;
		size_t* count_table = grayCountTable(impl_->r_matrix_);

		size_t count = 0;
		for (int i = 0; i < 256; i++) {
			if (count_table[i]) {
				count += count_table[i];
				if (count > min_thre) {
					min_gray = i;
					break;
				}
			}
		}
		count = 0;
		for (int i = 255; i >= 0; i--) {
			if (count_table[i]) {
				count += count_table[i];
				if (count > max_thre) {
					max_gray = i;
					break;
				}
			}
		}
		uint8_t map_table[256]{0};
		for (int i = 0; i < 256; i++) {
			if (i < min_gray) {
				map_table[i] = 0;
			} else if (i >= max_gray) {
				map_table[i] = 255;
			} else {
				map_table[i] = static_cast<uint8_t>((i - min_gray) * 255.0 / max_gray);
			}
		}
		mapGrayImage(impl_->r_matrix_, map_table, impl_->size());

		delete[] count_table;
		return *this;
	}

	auto ImageProcessor::edgeExtra(SharpenModel model, int index) -> ImageProcessor& {
		auto mode = checkModelIndex(model, index);

		auto sharpen_lam = [this](Image* to_sharpen, const ModelMat& model_m) -> Image* {
			to_sharpen->impl()->r_matrix_ = imageSharpen(impl_->r_matrix_, model_m);
			if (impl_->bpp_ != 8) {
				to_sharpen->impl()->g_matrix_ = imageSharpen(impl_->g_matrix_, model_m);
				to_sharpen->impl()->b_matrix_ = imageSharpen(impl_->b_matrix_, model_m);
			}
			return to_sharpen;
		};

		switch (mode) {
		case NORMAL:
			sharpen_lam(img_, ModelMap::INSTANCE(model, index));
			return *this;
		case MIX:
			sharpen_lam(img_, ModelMap::INSTANCE(model, SHARPEN_X));
			Image copy = *img_;
			return addWeighted(
				sharpen_lam(&copy, ModelMap::INSTANCE(model, SHARPEN_Y)),
				0.5, 0.5
			);
		}
		throw std::exception("ImageSharpen: Sharpen wrong!");
	}

	auto ImageProcessor::sharpen(SharpenModel model, float strength, int index) -> ImageProcessor& {
		Image copy = *img_;
		return addWeighted(ImageProcessor(copy).edgeExtra(model, index).get(), 1, strength);
	}

	auto ImageProcessor::canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		EdgeInfo e_info;
		edgeCanny(e_info, l_threshold, h_threshold, kernelsize, l2_gradient);
		return *this;
	}

	auto ImageProcessor::edgeCanny(EdgeInfo& edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		getEdgeInfo(&edgeinfo, gaussianFilter(kernelsize, 1.5).get(), l2_gradient);
		nonMaxSuppression(&edgeinfo);
		twoThreshold(&edgeinfo, l_threshold, h_threshold);
		return *this;
	}

	auto ImageProcessor::edgeCanny(EdgeInfo* edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		return edgeCanny(*edgeinfo, l_threshold, h_threshold, kernelsize, l2_gradient);
	}

	auto ImageProcessor::tiltCorrection(size_t zoomsize, bool copy) -> ImageProcessor& {
		ImageProcessor cpy_pro(copy ? img_->copy() : img_);
		if (cpy_pro.impl_->bpp_ != 8) {
			cpy_pro.toGray();
		}
		cpy_pro.canny(50, 100);
		zoomsize = zoomsize ? zoomsize : std::ranges::min(impl_->width(), impl_->height());
		auto hough = lineExtra(cpy_pro.impl_->r_matrix_, zoomsize);
#ifdef _DEBUG
			for (size_t i = 0; i < hough->size; i++) {
				double theta = hough->max_thetas[i];
				double radius = hough->max_redius[i];
				printf("¦È%lld=%lf¡ã, r%lld=%lf", i, theta / std::numbers::pi * 180, i, radius);
				drawLine(*impl_, radius, theta, 0xFF0099, 4);
			}
#endif
		if (hough->size) {
			const double pidiv2 = std::numbers::pi_v<double> / 2;
			auto k_theta = hough->max_thetas[0] > pidiv2
				               ? hough->max_thetas[0] - pidiv2
				               : hough->max_thetas[0] + pidiv2;
			auto theta = k_theta <= pidiv2 ? k_theta : std::numbers::pi - k_theta;
			rotate(*img_, theta, k_theta <= pidiv2);
		}

		delete hough;
		return *this;
	}

	auto ImageProcessor::addWeighted(const Image& other, float w1, float w2, uint8_t r) -> ImageProcessor& {
		if (impl_->bpp_ != other.impl()->bpp_) {
			throw std::exception("AddWeighted: Non-uniform bpp!");
		}
		if (impl_->width() != other.width() || impl_->height() != other.height()) {
			throw std::exception("AddWeighted: Non-uniform size!");
		}

		imageAddWeighted(impl_->r_matrix_, w1, other.impl()->r_matrix_, w2, r);
		if (impl_->bpp_ != 8) {
			imageAddWeighted(impl_->g_matrix_, w1, other.impl()->g_matrix_, w2, r);
			imageAddWeighted(impl_->b_matrix_, w1, other.impl()->b_matrix_, w2, r);
		}

		return *this;
	}

	auto ImageProcessor::addWeighted(const Image* other, float w1, float w2, uint8_t r) -> ImageProcessor& {
		return addWeighted(*other, w1, w2, r);
	}

	auto ImageProcessor::get() const -> Image& {
		return *img_;
	}

	auto ImageProcessor::checkModelIndex(SharpenModel model, int index) -> SharpenMode {
		switch (model) {
		case LOG:
		case LAPLACIAN:
			if (index > 1 || index < 0) {
				throw std::exception("ModelIndexCheck: Index out of range!");
			}
			return NORMAL;
		case ROBERTS:
		case PREWITT:
		case SOBEL:
			if (index > 2 || index < 0) {
				throw std::exception("ModelIndexCheck: Index out of range!");
			}
			return index ? NORMAL : MIX;
		}
		throw std::exception("ModelIndexCheck: Model wrong!");
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

	auto twoThreshold(const EdgeInfo* e_info, uint8_t l_threshold, uint8_t h_threshold) -> void {
		auto impl = e_info->edge->impl();
		auto r_data = impl->r_matrix_.data();
		auto g_data = impl->g_matrix_.data();
		auto b_data = impl->b_matrix_.data();
		memcpy(r_data, std::shared_ptr<uint8_t[]>(
			       hostTwoThreshold(r_data, impl->width(), impl->height(), l_threshold, h_threshold)
		       ).get(), impl->size());
		if (impl->bpp_ != 8) {
			memcpy(g_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(g_data, impl->width(), impl->height(), l_threshold, h_threshold)
			       ).get(), impl->size());
			memcpy(b_data, std::shared_ptr<uint8_t[]>(
				       hostTwoThreshold(b_data, impl->width(), impl->height(), l_threshold, h_threshold)
			       ).get(), impl->size());
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
		auto impl = img.impl();

		ImageProcessor(img).toRGB();
		hostDrawLine(impl->r_matrix_.data(), width, height, radius, theta, red, thickness);
		hostDrawLine(impl->g_matrix_.data(), width, height, radius, theta, green, thickness);
		hostDrawLine(impl->b_matrix_.data(), width, height, radius, theta, blue, thickness);
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
