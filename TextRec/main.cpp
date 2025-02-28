import std.compat;
import textRec;
import <FreeImage.h>;

#include "main.h"
#include <cstdio>

using namespace witcher_pic;

auto run(const char* pic_name) -> void;

auto main(int ARGV, char* ARGC[]) -> int {
	FreeImage_Initialise();
	std::string cmd(ARGC[1]);
	// 命令判断
	if (cmd == "recg") {
		CHECK_CMD(cmd, 1, 1)
		auto pic_name = FIXED_ARGC(0);
		// 文字识别
		try {
			// recognizeText(pic_name);
			run(pic_name);
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		// end
	} else if (cmd == "mix") {
		CHECK_CMD(cmd, 2, 2)
		// 图像融合
		try {
			std::println("正在融合...");
			mixImage(FIXED_ARGC(0), 0.5, FIXED_ARGC(1), 0.5);
			std::println("融合成功!");
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		// end
	} else {
		fprintf(stderr, "command \"%s\" undefined!", cmd.data());
	}

	FreeImage_DeInitialise();
	return 0;
}

auto run(const char* pic_name) -> void {
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);

	if (!FreeImage_FIFSupportsReading(format)) {
		throw std::exception("该图像不支持读取");
	}
	std::println("图像已读取");
	// gpuDeviceInfo();
	Image& img = *loadImage(pic_name);
	std::println("图片大小: {} * {}", img.width(), img.height());
	
	saveImage("dist/canny.bmp", img.toGray().canny(40, 70)->edge);
	std::println("{}", img.bpp());
}
