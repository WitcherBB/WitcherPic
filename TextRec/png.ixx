export module png;
export import :deflate;
import std.compat;
import <stdint.h>;
import wiioStream;

using namespace wstream;

namespace witcher_pic {
	export struct PNG;
	// 函数的导出声明
	export void fromPNG(const char* pic_name, PNG& picture);
	export void fromPNG(const char* pic_name, PNG* pic);

	// 内部函数的声明
	void readDataChunk(const ByteBuffer& buf, PNG& picture);
	inline std::string toTypeCode(uint32_t code);

	struct PNG {
		using rgba = uint32_t;

		struct Pixel {
			uint32_t x, y;
			rgba color;
		} * pixels;

		uint32_t width = 0, height = 0;
		bool done = false;
		uint8_t bit_depth;
		uint8_t color_type;
		uint8_t compression;
		uint8_t filter;
		uint8_t interlace;
	};

	const unsigned char PNG_HEAD[8] = {137, 80, 78, 71, 13, 10, 26, 10};

	void fromPNG(const char* pic_name, PNG& picture) {
		std::ifstream pic(pic_name, std::ios::in | std::ios::binary);
		// 获取文件长度
		pic.seekg(0, pic.end);
		int64_t len = pic.tellg();
		pic.seekg(0, pic.beg);
		// 读取文件
		char* buffer = new char[len];
		pic.read(buffer, len);
		ByteBuffer data(buffer, len);
		// 判断是否为PNG文件
		for (int i = 0; i < 8; i++) {
			uint8_t ch = 0;
			data.read(ch);
			if (ch != PNG_HEAD[i]) {
				std::cout << "This image is not PNG\n";
				return;
			}
		}
		// 开始分析内容
		// 文件头 IHDR
		while (!picture.done) {
			readDataChunk(data, picture);
		}
	}

	void fromPNG(const char* pic_name, PNG* pic) {
		fromPNG(pic_name, *pic);
	}


	auto readDataChunk(const ByteBuffer& buf, PNG& picture) -> void {
		uint32_t data_len = buf.readUInt32();
		
		std::string chunk_type = toTypeCode(buf.readUInt32());
		if (chunk_type == "IHDR") {
			buf.readUInt32(picture.width);
			buf.readUInt32(picture.height);
			buf.read(picture.bit_depth);
			buf.read(picture.color_type);
			buf.read(picture.compression);
			buf.read(picture.filter);
			buf.read(picture.interlace);
			std::println("位深度: {}", picture.bit_depth);
			std::println("颜色类型: {}", picture.color_type);
			std::println("压缩方法: {}", picture.compression);
			std::println("过滤方法: {}", picture.filter);
			std::println("隔行扫描: {}", picture.interlace);
		} else if (chunk_type == "PLTE") {
			std::println("调色板的长度: {}", data_len);
			buf.move(data_len);
		} else if (chunk_type == "IDAT") {
			std::println("这是IDAT");
			uint8_t cmf = buf.read();
			uint8_t flg = buf.read();
			size_t win_size = 0x01 << ((cmf & 0xF0) >> 4);
			bool if_use_dict = flg & 0x20;
			uint8_t flevel = (flg & 0xC0) >> 6;
			std::println("是否采用Deflate压缩： {}", (cmf & 0x0F) == 0x08);
			std::println("窗口大小: {}", win_size);
			std::println("校验结果: {}", !(((cmf << 8) + flg) % 31));
			std::println("预置字典: {}", if_use_dict);
			std::println("压缩级别: {}", flevel);
			buf.move(data_len - 2);
		}
		else if (chunk_type == "IEND") {
			picture.done = true;
		} else {
			buf.move(data_len);
		}
		buf.readUInt32();
	}

	auto toTypeCode(uint32_t code) -> std::string {
		return std::string({
			static_cast<char>((code >> 24) & 0xFF),
			static_cast<char>((code >> 16) & 0xFF),
			static_cast<char>((code >> 8) & 0xFF),
			static_cast<char>((code) & 0xFF)
		});
	}
}
