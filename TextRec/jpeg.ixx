export module jpeg;
import std.compat;
import <stdint.h>;
import wiioStream;

using namespace wstream;

namespace witcher_pic {
	export class JPEG;
	// 导出的函数

	// 不导出的函数

	class JPEG {
	public:
		using rgb = std::array<uint8_t, 3>;
		using Mat = std::valarray<std::valarray<rgb>>;
	private:
		// 外面是列，里面是行
		Mat mat_ = Mat();
		uint32_t width_ = 0, height_ = 0;
	};

	export auto decodeJPEG(const char* pic_dir, JPEG& jpeg) -> void {
		std::ifstream f_pic(pic_dir, std::ifstream::in | std::ios::binary);
		// 获取文件长度
		f_pic.seekg(0, f_pic.end);
		int64_t len = f_pic.tellg();
		f_pic.seekg(0, f_pic.beg);
		// 读取文件
		char* buffer = new char[len];
		f_pic.read(buffer, len);
		ByteBuffer data(buffer, len);
		// 分析JPEG
		uint16_t s1 = data.readUInt16();
		uint16_t s2 = data.readUInt16();
		std::printf("%X\n%X", s1, s2);
	}
}
