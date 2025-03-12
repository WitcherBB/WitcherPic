#include "main.h"
#include <FreeImage.h>
import witcherPic;

using namespace witcher_pic;

auto run(const char* pic_name) -> void;

auto main(int ARGV, char* ARGC[]) -> int {
	FreeImage_Initialise();
	witcher_pic::init();
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
			printf("mixing...\n");
			mixImage(FIXED_ARGC(0), 0.5, FIXED_ARGC(1), 0.5);
			printf("mixed seccessfully!\n");
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
		throw std::exception("This image is not supported to read.");
	}
	printf("Image has been read.\n");
	// gpuDeviceInfo();
	Image& img = *loadImage(pic_name);
	printf("Image Size: %u * %u\n", img.width(), img.height());

	ImageProcessor<true> processor1(img);
	ImageProcessor<true> processor2(img);
	saveImage("dist/canny.bmp", processor1.tiltCorrection(999).get());
	saveImage("dist/edge.png", processor2.toGray().canny(50, 100, 5, true).toOtsuBinary().get());
	printf("%d", img.bpp());
}
