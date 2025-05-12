#pragma once
#include <stddef.h>
#include <stdint.h>

namespace witcher_pic {
    struct HoughInfo {
		double* max_radius;
		double* max_thetas;
		size_t size;

		~HoughInfo();
	};
}
