export module textRec;
export import :util;
import std.compat;
import <FreeImage.h>;
import "textRec_types.h";

#pragma comment(lib, "FreeImage.lib")

namespace witcher_pic {
	export auto recognizeText(const char* pic_name) -> void;
	export auto mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r = 0) -> void;
	export auto loadImage(const char* pic_name) -> Image*;
	export auto saveImage(const char* filename, const Image& image) -> void;
	export auto saveImage(const char* filename, const Image* image) -> void;
}

namespace witcher_pic {
	// 此函数会抛异常，请使用 try catch 语句包围
	auto recognizeText(const char* pic_name) -> void {
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);

		if (!FreeImage_FIFSupportsReading(format)) {
			throw std::exception("该图像不支持读取");
		}
		std::println("图像已读取");

		Image* img = loadImage(pic_name);
		std::println("图片大小: {} * {}", img->width(), img->height());

		
	}

	auto mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r) -> void {
		Image* img1 = loadImage(pic_name1);
		Image* img2 = loadImage(pic_name2);

		Image& result = Image::addWeighted(*img1, 0.5, *img2, 0.5, r);
		saveImage("dist/mixedrgb.bmp", result);

		delete img1;
		delete img2;
	}

	auto loadImage(const char* pic_name) -> Image* {
		auto format = FreeImage_GetFileType(pic_name);
		// 从堆里创建
		FIBITMAP* bitmap = FreeImage_Load(format, pic_name);
		Finally f0([&bitmap]() {
			FreeImage_Unload(bitmap);
		});
		FreeImage_Initialise();
		unsigned bpp = FreeImage_GetBPP(bitmap);
		FREE_IMAGE_TYPE file_type = FreeImage_GetImageType(bitmap);
		BYTE* bytes = FreeImage_GetBits(bitmap);
		unsigned width = FreeImage_GetWidth(bitmap);
		unsigned height = FreeImage_GetHeight(bitmap);
		unsigned pitch = FreeImage_GetPitch(bitmap);
		if (file_type != FIT_BITMAP) {
			throw std::exception("该图片类型不是位图");
		}
#ifdef _DEBUG
		std::println("{}", bpp);
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
			throw std::exception("图片加载错误");
		}

#ifdef _DEBUG
		
#endif

		return img;
	}

	auto saveImage(const char* filename, const Image& image) -> void {
		unsigned width = image.width();
		unsigned height = image.height();
		int bpp = image.bpp();

		FIBITMAP* bitmap = FreeImage_Allocate(static_cast<int>(width), static_cast<int>(height), bpp);
		if (!bitmap) {
			FreeImage_DeInitialise();
			throw std::exception("位图创建失败");
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

	auto saveImage(const char* filename, const Image* image) -> void {
		saveImage(filename, *image);
	}
}
