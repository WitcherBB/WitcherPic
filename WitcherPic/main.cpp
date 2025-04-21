#include "main.h"
#include <FreeImage.h>
#include <witcherPic.h>

using namespace witcher_pic;

auto run(const char* pic_name) -> void;

auto main(int ARGV, char* ARGC[]) -> int {
	witcherpic_init();
	std::string cmd(ARGC[1]);
	// 命令判断
	if (cmd == "recg") {
		CHECK_CMD(cmd.c_str(), 1, 1)
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
		CHECK_CMD(cmd.c_str(), 2, 2)
		// 图像融合
		try {
			printf("mixing...\n");
			witcherpic_mixImage(FIXED_ARGC(0), 0.5, FIXED_ARGC(1), 0.5);
			printf("mixed seccessfully!\n");
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		// end
	} else {
		fprintf(stderr, "command \"%s\" undefined!", cmd.data());
	}

	witcherpic_deinit();
	return 0;
}

auto run(const char* pic_name) -> void {
	gpuDeviceInfo();
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(pic_name);

	if (!FreeImage_FIFSupportsReading(format)) {
		throw std::runtime_error("This image is not supported to read.");
	}
	printf("Image has been read.\n");
	Image& img = *witcherpic_loadImage(pic_name);
	printf("Image Size: %u * %u\n", img.width(), img.height());

	ImageProcessor processor1(img);
	ImageProcessor processor2(img.copy());
	witcherpic_refSaveImage("dist/canny.bmp", processor1.tiltCorrection(499, true).get());
	witcherpic_refSaveImage("dist/edge.png", processor2.toOtsuBinary().get());
	printf("%d", img.bpp());
}
