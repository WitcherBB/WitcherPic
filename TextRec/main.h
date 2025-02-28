#pragma once

#define ARGV argv
#define ARGC argc
#define FIXED_ARGC(index) ARGC[##index## + 2]

#define CHECK_CMD(cmd, min, max) if (ARGV> (##max## + 2)) { \
		std::println("{}: ��������",##cmd##); \
		return 1; \
	} \
	if (ARGV< (##min## + 2)) { \
		std::println("{}: ��������",##cmd##); \
		return 2; \
	}