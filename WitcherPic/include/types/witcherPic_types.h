#pragma once

namespace witcher_pic {
	enum FilterType {
		CONV,
		MEDIAN
	};

	struct FilterInfo {
		int source_w;
		int source_h;
		int rcx;
		int rcy;
		int model_w;
		int model_h;
		FilterType type;

		FilterInfo(int _sw, int _sh, int _rcx, int _rcy, int _mw, int _mh, FilterType ty);
	};
}
