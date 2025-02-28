export module textRec:util;
import std.compat;
import <Eigen/Dense>;
import "textRec_types.h";

using namespace Eigen;

#define SHARPEN_X 1
#define SHARPEN_Y 2

template <typename F>
class Finally {
	F func_;

public:
	Finally(F&& func) : func_(std::move(func)) {
	}

	~Finally() {
		func_();
	}
};

namespace witcher_pic {
	class Image;
	struct EdgeInfo;

	export template <typename T>
	using GenMatrix = Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>;
	export using ImgMat = GenMatrix<uint8_t>;
	export using ModelMat = GenMatrix<float>;

	using rgba = uint32_t;

	enum SharpenModel:uint8_t {
		LAPLACIAN,
		SOBEL,
		ROBERTS,
		PREWITT,
		LOG
	};

	export class ModelMap {
	public:
		class SModelPairHash {
		public:
			auto operator()(const std::pair<SharpenModel, int>& pair) const noexcept -> size_t {
				return std::hash<size_t>()(static_cast<size_t>(pair.first) | static_cast<size_t>(pair.second) << 8);
			}
		};

		using base_map = std::unordered_map<std::pair<SharpenModel, int>, ModelMat*, SModelPairHash>;

		ModelMap() = delete;
		ModelMap(const ModelMap&) = delete;
		ModelMap(ModelMap&&) = delete;

		auto operator()(SharpenModel model, int index) const -> const ModelMat& {
			return *map_.at(std::pair(model, index));
		}

		static const ModelMap INSTANCE;

	private:
		ModelMap(std::initializer_list<base_map::value_type> i_list) : map_(i_list) {
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

	auto imgFilter(const ImgMat& source, const ModelMat& model, int rcx,
	               int rcy, FilterType type = CONV) -> ImgMat;
	auto grayCountTable(const ImgMat& source) -> size_t*;
	auto mapGrayImage(ImgMat& target, const ImgMat& source, uint8_t* map_table,
	                  size_t size) -> void;
	auto imageSharpen(const ImgMat& source, const ModelMat& model) -> ImgMat;
	auto imageAddWeighted(const ImgMat& source1, float weight1, const ImgMat& source2, float weight2,
	                      uint8_t r) -> ImgMat;
	auto gaussianKernel(ModelMat& model, float sigma) -> void;
	auto insertData(const uint8_t* const* data, size_t datasize, int count) -> uint8_t*;
	auto getEdgeInfo(const Image* img, bool l2_gradient) -> EdgeInfo*;
	auto nonMaxSuppression(const EdgeInfo* e_info) -> void;
}

namespace witcher_pic {
	class Image {
		friend auto getEdgeInfo(const Image* img, bool l2_gradient) -> EdgeInfo*;
		friend auto nonMaxSuppression(const EdgeInfo* e_info) -> void;

		enum SharpenMode {
			NORMAL, MIX
		};

	public:
		Image(unsigned width, unsigned height, int bpp);
		Image(const Image& mat);
		auto resize(unsigned width, unsigned height) -> void;
		auto resizeLike(const Image& other) -> void;
		auto resizeLike(const Image* other) -> void;
		auto putPixel(unsigned x, unsigned y, rgba color) -> void;
		auto putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void;
		auto filter(const ModelMat& model, int rcx, int rcy, FilterType type) const -> Image*;
		auto averFilter(unsigned size) const -> Image*;
		auto medianFilter(unsigned size) const -> Image*;
		auto gaussianFilter(unsigned size, float sigma) const -> Image*;
		auto data() const -> uint8_t*;
		auto toGray() const -> Image*;
		auto toBinary(uint8_t m) const -> Image*;
		auto toOtsuBinary() const -> Image*;
		auto grayEnhance(float min_rate = 0, float max_rate = 0) const -> Image*;
		auto edgeExtra(SharpenModel model, int index = 0) const -> Image*;
		auto sharpen(SharpenModel model, float strength, int index = 0) const -> Image*;
		auto canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize = 3,
		           bool l2_gradient = false) const -> EdgeInfo*;

		auto width() const -> unsigned;
		auto height() const -> unsigned;
		auto size() const -> size_t;
		auto bpp() const -> int;

		auto operator()(unsigned x, unsigned y) const -> rgba;

		static auto addWeighted(Image* s1, float w1, Image* s2, float w2, uint8_t r = 0) -> Image*;

	private:
		[[nodiscard]] static auto checkModelIndex(SharpenModel model, int index) -> SharpenMode;

	protected:
		ImgMat r_matrix_;
		ImgMat g_matrix_;
		ImgMat b_matrix_;
		ImgMat a_matrix_;
		const int bpp_;
	};

	struct EdgeInfo {
		using EdgeDirMat = GenMatrix<int>;

		Image* edge;
		EdgeDirMat* r_dir;
		EdgeDirMat* g_dir;
		EdgeDirMat* b_dir;

		~EdgeInfo() {
			delete edge;
			delete r_dir;
			delete g_dir;
			delete b_dir;
		}
	};

	Image::Image(unsigned width, unsigned height, int bpp): bpp_(bpp) {
		r_matrix_.resize(height, width);
		r_matrix_.fill(0u);
		g_matrix_.resize(height, width);
		g_matrix_.fill(0u);
		b_matrix_.resize(height, width);
		b_matrix_.fill(0u);
		a_matrix_.resize(height, width);
		a_matrix_.fill(255u);
	}

	Image::Image(const Image& mat): bpp_(mat.bpp_) {
		r_matrix_ = mat.r_matrix_;
		g_matrix_ = mat.g_matrix_;
		b_matrix_ = mat.b_matrix_;
		a_matrix_ = mat.a_matrix_;
	}

	auto Image::resize(unsigned width, unsigned height) -> void {
		r_matrix_.conservativeResize(height, width);
		g_matrix_.conservativeResize(height, width);
		b_matrix_.conservativeResize(height, width);
		a_matrix_.conservativeResize(height, width);
	}

	auto Image::resizeLike(const Image& other) -> void {
		r_matrix_.conservativeResizeLike(other.r_matrix_);
		g_matrix_.conservativeResizeLike(other.g_matrix_);
		b_matrix_.conservativeResizeLike(other.b_matrix_);
		a_matrix_.conservativeResizeLike(other.a_matrix_);
	}

	auto Image::resizeLike(const Image* other) -> void {
		resizeLike(*other);
	}

	auto Image::putPixel(unsigned x, unsigned y, rgba color) -> void {
		r_matrix_(y, x) = static_cast<uint8_t>((color >> 24) & 0xFF);
		g_matrix_(y, x) = static_cast<uint8_t>((color >> 16) & 0xFF);
		b_matrix_(y, x) = static_cast<uint8_t>((color >> 8) & 0xFF);
		a_matrix_(y, x) = static_cast<uint8_t>(color & 0xFF);
	}

	auto Image::putPixel(unsigned x, unsigned y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> void {
		r_matrix_(y, x) = r;
		g_matrix_(y, x) = g;
		b_matrix_(y, x) = b;
		a_matrix_(y, x) = a;
	}

	auto Image::filter(const ModelMat& model, int rcx, int rcy, FilterType type) const -> Image* {
		Image* img = new Image(width(), height(), bpp_);
		img->r_matrix_ = imgFilter(this->r_matrix_, model, rcx, rcy, type);
		if (bpp_ != 8) {
			img->g_matrix_ = imgFilter(this->g_matrix_, model, rcx, rcy, type);
			img->b_matrix_ = imgFilter(this->b_matrix_, model, rcx, rcy, type);
		}
		return img;
	}

	auto Image::averFilter(unsigned size) const -> Image* {
		if ((size + 1) % 2) {
			throw std::exception("高斯模板大小必须为奇数");
		}
		ModelMat model(size, size);
		model.fill(1.0 / (size * size));
		return filter(model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto Image::medianFilter(unsigned size) const -> Image* {
		if ((size + 1) % 2) {
			throw std::exception("midianFilter: 模板大小必须为奇数");
		}
		ModelMat model(size, size);
		model.fill(1);
		return filter(model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, MEDIAN);
	}

	auto Image::gaussianFilter(unsigned size, float sigma) const -> Image* {
		if ((size + 1) % 2) {
			throw std::exception("gaussianFilter: 模板大小必须为奇数");
		}

		ModelMat model(size, size);
		gaussianKernel(model, sigma);
		return filter(model, (static_cast<int>(size) - 1) / 2, (static_cast<int>(size) - 1) / 2, CONV);
	}

	auto Image::data() const -> uint8_t* {
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

	auto Image::toGray() const -> Image* {
		Image* img = new Image(width(), height(), 8);
		for (size_t i = 0; i < size(); i++) {
			uint8_t gray = static_cast<uint8_t>(
				static_cast<float>(r_matrix_(i)) * 0.299F +
				static_cast<float>(g_matrix_(i)) * 0.587F +
				static_cast<float>(b_matrix_(i)) * 0.114F
			);
			img->r_matrix_(i) = gray;
		}
		return img;
	}

	auto Image::toBinary(uint8_t m) const -> Image* {
		Image* img = new Image(width(), height(), 8);

		if (bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return img;
		}

		for (auto y = 0u; y < height(); ++y) {
			for (auto x = 0u; x < width(); ++x) {
				uint8_t gray = r_matrix_(y, x) >= m ? 255 : 0;
				img->r_matrix_(y, x) = gray;
			}
		}
		return img;
	}

	auto Image::toOtsuBinary() const -> Image* {
		if (bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return new Image{width(), height(), 8};
		}

		size_t* gray_table = grayCountTable(r_matrix_);
		double variance[256] = {0};
		auto img_size = size();
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

	auto Image::grayEnhance(float min_rate, float max_rate) const -> Image* {
		if (bpp_ != 8) {
			fprintf(stderr, "This image is not gray image.\n");
			return new Image{width(), height(), 8};
		}
		if (min_rate > 100 || max_rate > 100 || min_rate + max_rate > 100 || min_rate < 0 || max_rate < 0) {
			fprintf(stderr, "rate wrong!\n");
			return new Image{width(), height(), 8};
		}

		size_t min_thre = static_cast<size_t>(size() / 100.0 * min_rate);
		size_t max_thre = static_cast<size_t>(size() / 100.0 * max_rate);
		uint8_t min_gray = 0;
		uint8_t max_gray = 0;
		size_t* count_table = grayCountTable(r_matrix_);

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
		Image* img = new Image(width(), height(), bpp_);
		mapGrayImage(img->r_matrix_, r_matrix_, map_table, size());

		delete[] count_table;
		return img;
	}

	auto Image::edgeExtra(SharpenModel model, int index) const -> Image* {
		auto mode = checkModelIndex(model, index);

		auto sharpen_lam = [this](const ModelMat& model_m) -> Image* {
			Image* img = new Image(this->width(), this->height(), bpp_);
			img->r_matrix_ = imageSharpen(r_matrix_, model_m);
			if (bpp_ != 8) {
				img->g_matrix_ = imageSharpen(g_matrix_, model_m);
				img->b_matrix_ = imageSharpen(b_matrix_, model_m);
			}
			return img;
		};

		switch (mode) {
		case NORMAL:
			return sharpen_lam(ModelMap::INSTANCE(model, index));
		case MIX:
			Image* img = addWeighted(
				std::shared_ptr<Image>(sharpen_lam(ModelMap::INSTANCE(model, SHARPEN_X))).get(), 0.5,
				std::shared_ptr<Image>(sharpen_lam(ModelMap::INSTANCE(model, SHARPEN_Y))).get(), 0.5);
			return img;
		}
		throw std::exception("ImageSharpen: Sharpen wrong!");
	}

	auto Image::sharpen(SharpenModel model, float strength, int index) const -> Image* {
		Image copy = *this;
		Image* edge = edgeExtra(model, index);
		return addWeighted(&copy, 1, edge, strength);
	}

	auto Image::canny(uint8_t l_threshold, uint8_t h_threshold, unsigned kernelsize,
	                  bool l2_gradient) const -> EdgeInfo* {
		EdgeInfo* e_info = getEdgeInfo(std::shared_ptr<Image>(gaussianFilter(kernelsize, 1)).get(), l2_gradient);
		nonMaxSuppression(e_info);

		return e_info;
	}

	auto Image::width() const -> unsigned {
		return r_matrix_.cols();
	}

	auto Image::height() const -> unsigned {
		return r_matrix_.rows();
	}

	auto Image::size() const -> size_t {
		return width() * height();
	}

	auto Image::bpp() const -> int {
		return bpp_;
	}

	auto Image::operator()(unsigned x, unsigned y) const -> rgba {
		return static_cast<rgba>(
			r_matrix_(y, x) << 24 |
			g_matrix_(y, x) << 16 |
			b_matrix_(y, x) << 8 |
			a_matrix_(y, x)
		);
	}

	auto Image::checkModelIndex(SharpenModel model, int index) -> SharpenMode {
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

	auto Image::addWeighted(Image* s1, float w1, Image* s2, float w2, uint8_t r) -> Image* {
		if (s1->width() > s2->width() || s1->height() > s2->height()) {
			s2->resizeLike(*s1);
		} else if (s1->width() < s2->width() || s1->height() < s2->height()) {
			s1->resizeLike(*s2);
		}

		if (s1->bpp_ != s2->bpp_) {
			throw std::exception("AddWeighted: Non-uniform bpp");
		}

		Image* img = new Image(s1->width(), s1->height(), s1->bpp_);
		img->r_matrix_ = imageAddWeighted(s1->r_matrix_, w1, s2->r_matrix_, w2, r);
		if (s1->bpp_ != 8) {
			img->g_matrix_ = imageAddWeighted(s1->g_matrix_, w1, s2->g_matrix_, w2, r);
			img->b_matrix_ = imageAddWeighted(s1->b_matrix_, w1, s2->b_matrix_, w2, r);
		}

		return img;
	}
}
