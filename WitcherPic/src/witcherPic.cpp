#include "witcherPic_util.h"
#include "witcher_template.hpp"

#include <FreeImage.h>
#include <stdexcept>
#include <fmt/color.h>

namespace witcher_pic {
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
		case 32: {
			const uint8_t* bgra_data[4];
			bgra_data[0] = b_matrix_.data();
			bgra_data[1] = g_matrix_.data();
			bgra_data[2] = r_matrix_.data();
			bgra_data[3] = a_matrix_.data();
			return insertData(bgra_data, size(), 4);
		}
		case 24: {
			const uint8_t* bgr_data[3];
			bgr_data[0] = b_matrix_.data();
			bgr_data[1] = g_matrix_.data();
			bgr_data[2] = r_matrix_.data();
			return insertData(bgr_data, size(), 3);
		}
		case 8: {
			uint8_t* gray_data = new uint8_t[size()];
			memcpy(gray_data, r_matrix_.data(), size());
			return gray_data;
		}
		default:
			throw std::runtime_error("bpp wrong!");
		}
	}

    auto ImageImpl::normalData() const -> uint8_t *
    {
		switch (bpp_) {
		case 32: {
			auto img_size = this->size();
			uint8_t* img_data = new uint8_t[img_size * 4];
			memcpy(img_data + 0 * img_size, r_matrix_.data(), img_size);
			memcpy(img_data + 1 * img_size, g_matrix_.data(), img_size);
			memcpy(img_data + 2 * img_size, b_matrix_.data(), img_size);
			memcpy(img_data + 3 * img_size, a_matrix_.data(), img_size);
			return img_data;
		}
		case 24: {
			auto img_size = this->size();
			uint8_t* img_data = new uint8_t[img_size * 3];
			memcpy(img_data + 0 * img_size, r_matrix_.data(), img_size);
			memcpy(img_data + 1 * img_size, g_matrix_.data(), img_size);
			memcpy(img_data + 2 * img_size, b_matrix_.data(), img_size);
			return img_data;
		}
		case 8: {
			uint8_t* gray_data = new uint8_t[size()];
			memcpy(gray_data, r_matrix_.data(), size());
			return gray_data;
		}
		default:
			throw std::runtime_error("bpp wrong!");
		}
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

	ImageProcessor::ImageProcessor(Image* img): img_(img) {
	}

	ImageProcessor::ImageProcessor(Image& img): img_(&img) {
	}

	auto ImageProcessor::averFilter(unsigned size) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::runtime_error("The Gaussian template size must be odd.");
		}
		ModelMat model(size, size);
		model.fill(1.0 / (size * size));
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto ImageProcessor::medianFilter(unsigned size) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::runtime_error("midianFilter: template size must be odd.");
		}
		ModelMat model(size, size);
		model.fill(1);
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, MEDIAN);
	}

	auto ImageProcessor::gaussianFilter(unsigned size, float sigma) -> ImageProcessor& {
		if ((size + 1) % 2) {
			throw std::runtime_error("gaussianFilter: template size must be odd.");
		}

		ModelMat model(size, size);
		gaussianKernel(model, sigma);
		return filter(*this, model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto ImageProcessor::toRGBA() -> ImageProcessor& {
		auto img_impl = impl();
		if (img_impl->bpp_ != 32) {
			img_impl->a_matrix_.fill(255);
			if (img_impl->bpp_ == 8) {
				img_impl->g_matrix_ = img_impl->r_matrix_;
				img_impl->b_matrix_ = img_impl->r_matrix_;
			}
		}
		img_impl->bpp_ = 32;
		return *this;
	}

	auto ImageProcessor::toRGB() -> ImageProcessor& {
		auto img_impl = impl();
		if (img_impl->bpp_ == 8) {
			img_impl->g_matrix_ = img_impl->r_matrix_;
			img_impl->b_matrix_ = img_impl->r_matrix_;
		}
		img_impl->bpp_ = 24;
		return *this;
	}

	auto ImageProcessor::toGray() -> ImageProcessor& {
		auto img_impl = impl();
		img_impl->bpp_ = 8;
		for (size_t i = 0; i < img_impl->size(); i++) {
			uint8_t gray = static_cast<uint8_t>(
				static_cast<float>(img_impl->r_matrix_(i)) * 0.299F +
				static_cast<float>(img_impl->g_matrix_(i)) * 0.587F +
				static_cast<float>(img_impl->b_matrix_(i)) * 0.114F
			);
			img_impl->r_matrix_(i) = gray;
		}
		return *this;
	}

	auto ImageProcessor::toBinary(uint8_t m) -> ImageProcessor& {
		auto img_impl = impl();
		if (img_impl->bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return *this;
		}

		for (auto y = 0u; y < img_impl->height(); ++y) {
			for (auto x = 0u; x < img_impl->width(); ++x) {
				img_impl->r_matrix_(y, x) = img_impl->r_matrix_(y, x) >= m ? 255 : 0;;
			}
		}
		return *this;
	}

	auto ImageProcessor::toOtsuBinary() -> ImageProcessor& {
		if (impl()->bpp_ != 8) {
			print(fg(fmt::color::dark_red), "This image is not gray image.\n");
			return *this;
		}

		auto m = calcThresholdWithOtsu(img_);
		return toBinary(m[0]);
	}

	auto ImageProcessor::grayEnhance(float min_rate, float max_rate) -> ImageProcessor& {
		auto img_impl = impl();
		if (img_impl->bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return *this;
		}
		if (min_rate > 100 || max_rate > 100 || min_rate + max_rate > 100 || min_rate < 0 || max_rate < 0) {
			fprintf(stderr, "rate wrong!\n");
			return *this;
		}

		size_t min_thre = static_cast<size_t>(img_impl->size() / 100.0 * min_rate);
		size_t max_thre = static_cast<size_t>(img_impl->size() / 100.0 * max_rate);
		uint8_t min_gray = 0;
		uint8_t max_gray = 0;
		size_t* count_table = grayCountTable(img_impl->r_matrix_);

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
		mapGrayImage(impl()->r_matrix_, map_table, impl()->size());

		delete[] count_table;
		return *this;
	}

	auto ImageProcessor::edgeExtra(CppSharpenModel model, int index) -> ImageProcessor& {
		auto img_impl = impl();
		auto mode = checkModelIndex(model, index);

		auto sharpen_lam = [&img_impl](Image* to_sharpen, const ModelMat& model_m) -> Image* {
			to_sharpen->impl()->r_matrix_ = imageSharpen(img_impl->r_matrix_, model_m);
			if (img_impl->bpp_ != 8) {
				to_sharpen->impl()->g_matrix_ = imageSharpen(img_impl->g_matrix_, model_m);
				to_sharpen->impl()->b_matrix_ = imageSharpen(img_impl->b_matrix_, model_m);
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
		throw std::runtime_error("ImageSharpen: Sharpen wrong!");
	}

	auto ImageProcessor::sharpen(CppSharpenModel model, float strength, int index) -> ImageProcessor& {
		Image copy = *img_;
		return addWeighted(ImageProcessor(copy).edgeExtra(model, index).get(), 1, strength);
	}

	auto ImageProcessor::canny(unsigned kernelsize, bool l2_gradient) -> ImageProcessor& {
		EdgeInfo e_info;
		getEdgeInfo(&e_info, gaussianFilter(kernelsize, 1.5).get(), l2_gradient);
		nonMaxSuppression(&e_info);
		auto h_thresholds = NumVector<uint8_t>(calcThresholdWithOtsu(e_info.edge), img_->bpp() == 8 ? 1 : 3);
		auto l_thresholds = h_thresholds / 2;
		twoThreshold(&e_info, l_thresholds.data(), h_thresholds.data());
		return *this;
	}

	auto ImageProcessor::canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		EdgeInfo e_info;
		canny(e_info, l_threshold, h_threshold, kernelsize, l2_gradient);
		return *this;
	}

	auto ImageProcessor::canny(EdgeInfo& edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		getEdgeInfo(&edgeinfo, gaussianFilter(kernelsize, 1.5).get(), l2_gradient);
		nonMaxSuppression(&edgeinfo);
		twoThreshold(&edgeinfo, &l_threshold, &h_threshold);
		return *this;
	}

	auto ImageProcessor::canny(EdgeInfo* edgeinfo, uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                           bool l2_gradient) -> ImageProcessor& {
		return canny(*edgeinfo, l_threshold, h_threshold, kernelsize, l2_gradient);
	}

	auto ImageProcessor::tiltCorrection(size_t zoomsize, bool copy, unsigned kermelsize, bool l2_gradient) -> ImageProcessor& {
		auto img_impl = impl();
		ImageProcessor cpy_pro(copy ? img_->copy() : img_);
		auto cpy_img_impl = cpy_pro.impl();
		if (cpy_img_impl->bpp_ != 8) {
			cpy_pro.toGray();
		}
		cpy_pro.canny(kermelsize, l2_gradient);
		zoomsize = zoomsize ? zoomsize : std::ranges::min(img_impl->width(), img_impl->height());
		auto hough = lineExtra(cpy_img_impl->r_matrix_, zoomsize);
#ifdef _DEBUG
			for (size_t i = 0; i < hough->size; i++) {
				double theta = hough->max_thetas[i];
				double radius = hough->max_redius[i];
				printf("theta%lld=%lf rad, r%lld=%lf", i, theta, i, radius);
				drawLine(*impl, radius, theta, 0xFF0099, 4);
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
		auto img_impl = impl();
		if (img_impl->bpp_ != other.impl()->bpp_) {
			throw std::runtime_error("AddWeighted: Non-uniform bpp!");
		}
		if (img_impl->width() != other.width() || img_impl->height() != other.height()) {
			throw std::runtime_error("AddWeighted: Non-uniform size!");
		}

		imageAddWeighted(img_impl->r_matrix_, w1, other.impl()->r_matrix_, w2, r);
		if (img_impl->bpp_ != 8) {
			imageAddWeighted(img_impl->g_matrix_, w1, other.impl()->g_matrix_, w2, r);
			imageAddWeighted(img_impl->b_matrix_, w1, other.impl()->b_matrix_, w2, r);
		}

		return *this;
	}

	auto ImageProcessor::addWeighted(const Image* other, float w1, float w2, uint8_t r) -> ImageProcessor& {
		return addWeighted(*other, w1, w2, r);
	}

	auto ImageProcessor::get() const -> Image& {
		return *img_;
	}

	auto ImageProcessor::checkModelIndex(CppSharpenModel model, int index) -> SharpenMode {
		switch (model) {
		case LOG:
		case LAPLACIAN:
			if (index > 1 || index < 0) {
				throw std::runtime_error("ModelIndexCheck: Index out of range!");
			}
			return NORMAL;
		case ROBERTS:
		case PREWITT:
		case SOBEL:
			if (index > 2 || index < 0) {
				throw std::runtime_error("ModelIndexCheck: Index out of range!");
			}
			return index ? NORMAL : MIX;
		}
		throw std::runtime_error("ModelIndexCheck: Model wrong!");
	}

	auto ImageProcessor::calcThresholdWithOtsu(const Image* img) -> uint8_t* {
		int depth = img->bpp() == 8 ? 1 : 3;
		uint8_t* thresholds = new uint8_t[depth];
		ImgMat** mats = img->bpp() == 8
			                ? new ImgMat*[1]{&img->impl()->r_matrix_}
			                : new ImgMat*[3]{&img->impl()->r_matrix_, &img->impl()->g_matrix_, &img->impl()->b_matrix_};
		auto pi = new double[256];
		for (int _i = 0; _i < depth; _i++) {
			size_t* gray_table = grayCountTable(*mats[_i]);
			double variance[256] = {0};
			auto img_size = img->impl()->size();
			uint8_t L = 0;
			for (unsigned i = 0; i <= 255; i++) {
				if (gray_table[i]) {
					L = i;
				}
				pi[i] = static_cast<double>(gray_table[i]) / img_size;
			}
			double u = 0;
			for (unsigned i = 0; i < L; ++i) {
				u += i * pi[i];
			}

			for (unsigned i = 1; i < L; i++) {
				double u1 = 0.0;
				auto p1 = 0.0;
				for (unsigned j = 0; j < i; j++) {
					p1 += pi[j];
					u1 += j * pi[j];
				}
				variance[i] = (u * p1 - u1) * (u * p1 - u1) / (p1 * (1 - p1));
			}

			for (unsigned i = 0; i <= L; ++i) {
				if (variance[thresholds[_i]] < variance[i]) {
					thresholds[_i] = i;
				}
			}
			delete[] gray_table;
		}
		delete[] pi;
		return thresholds;
	}

	auto ImageProcessor::impl() const -> ImageImpl* {
		return img_->impl();
	}
}

using namespace witcher_pic;

extern "C" {
    auto witcherpic_recognizeText(const char* pic_name) -> void {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);
    
        if (!FreeImage_FIFSupportsReading(format)) {
            throw std::runtime_error("This image is not supported to read.");
        }
        printf("Image has been read.\n");
    
        Image* img = witcherpic_loadImage(pic_name);
        printf("Image Size: %u * %u\n", img->width(), img->height());
    }
    
    auto witcherpic_mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r) -> void {
        Image* img1 = witcherpic_loadImage(pic_name1);
        Image* img2 = witcherpic_loadImage(pic_name2);
    
        Image& result = ImageProcessor(img1).addWeighted(*img2, 0.5, 0.5, r).get();
        witcherpic_refSaveImage("dist/mixedrgb.bmp", result);
    
        delete img1;
        delete img2;
    }
    
    auto witcherpic_refSaveImage(const char* filename, const Image& image) -> void {
        unsigned width = image.width();
        unsigned height = image.height();
        int bpp = image.bpp();
    
        FIBITMAP* bitmap = FreeImage_Allocate(static_cast<int>(width), static_cast<int>(height), bpp);
        if (!bitmap) {
            FreeImage_DeInitialise();
            throw std::runtime_error("Bitmap creation failed");
        }
    
        auto data = image.data();
        for (auto y = 0u; y < height; y++) {
            auto line = FreeImage_GetScanLine(bitmap, y);
            memcpy(line, data + y * width * bpp / 8, width * bpp / 8);
        }
        delete[] data;
    
        FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
        if (format == FIF_UNKNOWN) {
            format = FIF_BMP;
        }
        FreeImage_Save(format, bitmap, filename);
    
        FreeImage_Unload(bitmap);
    }
    
    auto witcherpic_ptrSaveImage(const char* filename, const Image* image) -> void {
        witcherpic_refSaveImage(filename, *image);
    }
    
    auto witcherpic_loadImage(const char* pic_name) -> Image* {
        auto format = FreeImage_GetFileType(pic_name);
        // 从堆里创建
        FIBITMAP* bitmap = FreeImage_Load(format, pic_name);
        Finally f0([&bitmap]() {
            FreeImage_Unload(bitmap);
        });
        
        unsigned bpp = FreeImage_GetBPP(bitmap);
        FREE_IMAGE_TYPE file_type = FreeImage_GetImageType(bitmap);
        BYTE* bytes = FreeImage_GetBits(bitmap);
        unsigned width = FreeImage_GetWidth(bitmap);
        unsigned height = FreeImage_GetHeight(bitmap);
        unsigned pitch = FreeImage_GetPitch(bitmap);
        if (file_type != FIT_BITMAP) {
            throw std::runtime_error("Type of this image is not bitmap.");
        }
    #ifdef _DEBUG
            printf("%d", bpp);
    #endif
    
        Image* img = new Image(width, height, bpp);
    
        // BGRA => RGBA
        if (bpp == 32) {
            for (auto x = 0; x < width; x++) {
                for (auto y = 0; y < height; y++) {
                    auto bPos = y * pitch + x * 4;
                    uint8_t r = bytes[bPos + FI_RGBA_RED];
                    uint8_t g = bytes[bPos + FI_RGBA_GREEN];
                    uint8_t b = bytes[bPos + FI_RGBA_BLUE];
                    uint8_t a = bytes[bPos + FI_RGBA_ALPHA];
                    img->putPixel(x, y, r, g, b, a);
                }
            }
        } else if (bpp == 24) {
            for (auto x = 0; x < width; x++) {
                for (auto y = 0; y < height; y++) {
                    auto bPos = y * pitch + x * 3;
                    uint8_t r = bytes[bPos + FI_RGBA_RED];
                    uint8_t g = bytes[bPos + FI_RGBA_GREEN];
                    uint8_t b = bytes[bPos + FI_RGBA_BLUE];
                    img->putPixel(x, y, r, g, b, 255);
                }
            }
        } else if (bpp == 8) {
            for (auto x = 0; x < width; x++) {
                for (auto y = 0; y < height; y++) {
                    auto bPos = y * pitch + x;
                    uint8_t r = bytes[bPos];
                    img->putPixel(x, y, r, r, r, 255);
                }
            }
        } else {
            throw std::runtime_error("Image loading failed");
        }
    
    #ifdef _DEBUG
    
    #endif
    
        return img;
    }
    }
    