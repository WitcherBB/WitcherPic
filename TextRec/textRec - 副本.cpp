export module textRec;
export import :util;
import std.compat;
import <FreeImage.h>;
import "textRec_types.h";

#pragma comment(lib, "FreeImage.lib")

namespace witcher_pic {
	export auto recognizeText(const char* pic_name) -> void;
	export auto mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r = 0) -> void;
	auto loadImage(const char* pic_name) -> Image*;
	auto saveImage(const char* filename, const Image* image) -> void;
}

namespace witcher_pic {
	auto recognizeText(const char* pic_name) -> void {
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);

		if (!FreeImage_FIFSupportsReading(format)) {
			throw std::exception("��ͼ��֧�ֶ�ȡ");
		}
		std::println("ͼ���Ѷ�ȡ");

		Image* img = loadImage(pic_name);
		std::println("ͼƬ��С: {} * {}", img->width(), img->height());

		// RGB
		// Image* rgb_g_img = img->gaussianFilter(3, 1.5);
		// saveImage("dist/rgbGauss.bmp", img);
		// saveImage("dist/rgbSharpen.bmp", img->sharpen(LOG, 0.06, 0));

		// GRAY
		saveImage("dist/origin.bmp", std::shared_ptr<Image>(img->toGray()).get());
		// EdgeInfo* canny = img->toGray()->canny(0, 0);
		// saveImage("dist/canny.bmp", canny->edge);
		// Image* g_img = img->toGray()->gaussianFilter(9, 1);
		// saveImage("dist/gray.bmp", g_img);
		// Image* ge_img = g_img->grayEnhance();
		// saveImage("dist/grayEnhanced.bmp", ge_img);
		// Image* b_img = ge_img->toOtsuBinary();
		// saveImage("dist/binary.bmp", b_img);
		// saveImage("dist/sharpen.bmp", g_img->sharpen(LAPLACIAN, 0.618, 1));
		delete img;

#ifdef _DEBUG
		// std::printf("ԭ����0x%X\n", img(600, 600));
		// std::printf("������0x%X\n", r_img(600, 600));
#endif
	}

	auto mixImage(const char* pic_name1, float w1, const char* pic_name2, float w2, uint8_t r) -> void {
		Image* img1 = loadImage(pic_name1);
		Image* img2 = loadImage(pic_name2);

		Image* result = Image::addWeighted(img1, 0.5, img2, 0.5, r);
		saveImage("dist/mixedrgb.bmp", result);

		delete img1;
		delete img2;
	}

	auto loadImage(const char* pic_name) -> Image* {
		auto format = FreeImage_GetFileType(pic_name);
		// �Ӷ��ﴴ��
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
			throw std::exception("��ͼƬ���Ͳ���λͼ");
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
			throw std::exception("ͼƬ���ش���");
		}

#ifdef _DEBUG
		
#endif

		return img;
	}

	auto saveImage(const char* filename, const Image* image) -> void {
		unsigned width = image->width();
		unsigned height = image->height();
		int bpp = image->bpp();

		FIBITMAP* bitmap = FreeImage_Allocate(static_cast<int>(width), static_cast<int>(height), bpp);
		if (!bitmap) {
			FreeImage_DeInitialise();
			throw std::exception("λͼ����ʧ��");
		}

		auto data = image->data();
#ifdef _DEBUG
		
#endif

		memcpy(FreeImage_GetBits(bitmap), data, width * height * bpp / 8);
		delete data;

		FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
		if (format == FIF_UNKNOWN) {
			format = FIF_BMP;
		}
		FreeImage_Save(format, bitmap, filename);

		FreeImage_Unload(bitmap);
	}
}
