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
	// �����ж�
	if (cmd == "recg") {
		CHECK_CMD(cmd, 1, 1)
		auto pic_name = FIXED_ARGC(0);
		// ����ʶ��
		try {
			// recognizeText(pic_name);
			run(pic_name);
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		// end
	} else if (cmd == "mix") {
		CHECK_CMD(cmd, 2, 2)
		// ͼ���ں�
		try {
			std::println("�����ں�...");
			mixImage(FIXED_ARGC(0), 0.5, FIXED_ARGC(1), 0.5);
			std::println("�ںϳɹ�!");
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
		throw std::exception("��ͼ��֧�ֶ�ȡ");
	}
	std::println("ͼ���Ѷ�ȡ");
	// gpuDeviceInfo();
	Image& img = *loadImage(pic_name);
	std::println("ͼƬ��С: {} * {}", img.width(), img.height());
	
	saveImage("dist/canny.bmp", img.toGray().canny(40, 70)->edge);
	std::println("{}", img.bpp());
}
