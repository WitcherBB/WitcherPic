#include "textRec_types.h"

namespace witcher_pic {
	FilterInfo::FilterInfo(int _sw, int _sh, int _rcx, int _rcy, int _mw, int _mh,
	                       FilterType ty): source_w(_sw),
	                                       source_h(_sh),
	                                       rcx(_rcx),
	                                       rcy(_rcy),
	                                       model_w(_mw),
	                                       model_h(_mh),
	                                       type(ty) {
	}
}
