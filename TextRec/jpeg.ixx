export module jpeg;
import std.compat;
import <stdint.h>;
import wiioStream;

using namespace wstream;

namespace witcher_pic {
	export class JPEG;
	// �����ĺ���

	// �������ĺ���

	class JPEG {
	public:
		using rgb = std::array<uint8_t, 3>;
		using Mat = std::valarray<std::valarray<rgb>>;
	private:
		// �������У���������
		Mat mat_ = Mat();
		uint32_t width_ = 0, height_ = 0;
	};

	export auto decodeJPEG(const char* pic_dir, JPEG& jpeg) -> void {
		std::ifstream f_pic(pic_dir, std::ifstream::in | std::ios::binary);
		// ��ȡ�ļ�����
		f_pic.seekg(0, f_pic.end);
		int64_t len = f_pic.tellg();
		f_pic.seekg(0, f_pic.beg);
		// ��ȡ�ļ�
		char* buffer = new char[len];
		f_pic.read(buffer, len);
		ByteBuffer data(buffer, len);
		// ����JPEG
		uint16_t s1 = data.readUInt16();
		uint16_t s2 = data.readUInt16();
		std::printf("%X\n%X", s1, s2);
	}
}
