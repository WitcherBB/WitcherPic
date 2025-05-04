#include "main.h"
#include <FreeImage.h>
#include <witcherPic.h>
#include <fmt/core.h>
#include <cmath>
#include <ctime>
#include <functional>

using namespace witcher_pic;

auto test(const char* arg) -> void;

template<typename T_R, typename... Args>
auto costTimeMs(std::function<T_R(Args...)> func, T_R& res, Args... args) -> std::clock_t {
	auto st = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	res = func(std::forward<Args>(args)...);
	auto et = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	return et - st;
}

template<typename T_R>
auto costTimeMs(std::function<T_R()> func, T_R& res) -> std::clock_t {
	auto st = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	res = func();
	auto et = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	return et - st;
}

template<typename... Args>
auto costTimeMs(std::function<void(Args...)> func, Args... args) -> std::clock_t {
	auto st = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	func(std::forward<Args>(args)...);
	auto et = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	return et - st;
}

auto costTimeMs(std::function<void()> func) -> std::clock_t {
	auto st = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	func();
	auto et = static_cast<std::clock_t>(static_cast<long double>(std::clock()) / CLOCKS_PER_SEC * 1000);
	return et - st;
}

auto main(int ARGV, char* ARGC[]) -> int {
	witcherpic_init();
	std::string cmd(ARGC[1]);
	// 命令判断
	if (cmd == "recg") {
		CHECK_CMD(cmd.c_str(), 1, 1)
		auto pic_name = FIXED_ARGC(0);
		// 文字识别
		try {
			witcherpic_recognizeText(pic_name);
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
	} else if (cmd == "test") {
		CHECK_CMD(cmd.c_str(), 1, 1)
		auto pic_name = FIXED_ARGC(0);
		try {
			fmt::print("test 耗时: {}ms\n", costTimeMs(std::function(test), const_cast<const char*>(pic_name)));
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	} else {
		fprintf(stderr, "command \"%s\" undefined!", cmd.data());
	}

	witcherpic_deinit();
	return 0;
}

auto test(const char* arg) -> void {
	// gpuDeviceInfo();
	// FREE_IMAGE_FORMAT format = FreeImage_GetFileType(arg);

	// if (!FreeImage_FIFSupportsReading(format)) {
	// 	throw std::runtime_error("This image is not supported to read.");
	// }
	// printf("Image has been read.\n");
	// Image& img = *witcherpic_loadImage(arg);
	// printf("Image Size: %u * %u\n", img.width(), img.height());

	// ImageProcessor processor1(img);
	// ImageProcessor processor2(img.copy());
	// // witcherpic_refSaveImage("dist/canny.png", processor1.tiltCorrection(499, true).get());
	// fmt::print("process 耗时: {}ms\n", costTimeMs([&]() -> void {
	// 	processor2.resize(-1, 512, CppResizeMode::BICUBIC);
	// }));
	// witcherpic_refSaveImage("dist/edge.png", processor2.get());
	// printf("%d\n", img.bpp());
	auto func = [](int x) -> int {
		return x;
	};
	int result = func(std::stoi(arg));
	short a = 0xFFFE;
	printf("%d %d 0x%X\n", sizeof(short), a, a);
}
