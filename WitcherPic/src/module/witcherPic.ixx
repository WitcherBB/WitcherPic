module;
#include "FreeImage.h"

#include <cstdio>
#include <exception>

#include "cudaWitcherPic.h"
export module witcherPic;
export import :util;

using namespace witcher_pic;

extern "C" {
export auto witcherpic_init() -> void;
export auto witcherpic_recognizeText(const char* pic_name) -> void;
export auto witcherpic_mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2,
                                 uint8_t r = 0) -> void;
export auto witcherpic_loadImage(const char* pic_name) -> Image*;
export auto witcherpic_refSaveImage(const char* filename, const Image& image) -> void;
export auto witcherpic_ptrSaveImage(const char* filename, const Image* image) -> void;
}

extern "C" {
auto witcherpic_recognizeText(const char* pic_name) -> void {
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);

	if (!FreeImage_FIFSupportsReading(format)) {
		throw std::exception("This image is not supported to read.");
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
		throw std::exception("Bitmap creation failed");
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
	FreeImage_Initialise();
	unsigned bpp = FreeImage_GetBPP(bitmap);
	FREE_IMAGE_TYPE file_type = FreeImage_GetImageType(bitmap);
	BYTE* bytes = FreeImage_GetBits(bitmap);
	unsigned width = FreeImage_GetWidth(bitmap);
	unsigned height = FreeImage_GetHeight(bitmap);
	unsigned pitch = FreeImage_GetPitch(bitmap);
	if (file_type != FIT_BITMAP) {
		throw std::exception("Type of this image is not bitmap.");
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
		throw std::exception("Image loading failed");
	}

#ifdef _DEBUG

#endif

	return img;
}
}
