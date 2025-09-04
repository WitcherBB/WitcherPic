#pragma once

#include <cstdio>
#include <exception>
#include <iostream>

#define ARGV argv
#define ARGC argc
#define FIXED_ARGC(index) ARGC[index + 2]

#define CHECK_CMD(cmd, min, max) if (ARGV> (max + 2)) { \
		printf("%s: param too many\n", cmd); \
		return 1; \
	} \
	if (ARGV< (min + 2)) { \
		printf("%s: param too little\n", cmd); \
		return 2; \
	}
