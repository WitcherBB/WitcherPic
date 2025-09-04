#include <vector>

#define DEFINE_OPERATOR(op) \
	template<typename At> requires std::is_arithmetic_v<At>\
	auto operator op(const std::vector<At>& another) const -> NumVector {\
		NumVector result(base::size());\
		for (size_t i = 0; i < base::size(); ++i) {\
			result[i] = (*this)[i] op another[i];\
		}\
		return result;\
	}\
	template<typename An> requires std::is_arithmetic_v<An>\
	auto operator op(const An& another) const -> NumVector {\
		NumVector result(base::size());\
		for (size_t i = 0; i < base::size(); ++i) {\
			result[i] = (*this)[i] op another;\
		}\
		return result;\
	}

#define DEFINE_ASSIGN_OPERATOR(op) \
	template<typename At> requires std::is_arithmetic_v<At>\
	auto operator op(const std::vector<At>& another) const -> NumVector {\
		for (size_t i = 0; i < base::size(); ++i) {\
			(*this)[i] op another[i];\
		}\
		return *this;\
	}\
	template<typename An> requires std::is_arithmetic_v<An>\
	auto operator op(const An& another) const -> NumVector {\
		for (size_t i = 0; i < base::size(); ++i) {\
			(*this)[i] op another;\
		}\
		return *this;\
	}

	template <class Ty, class Alloc = std::allocator<Ty>>
	class NumVector : public std::vector<Ty, Alloc> {
	public:
		static_assert(std::is_arithmetic_v<Ty>, "Ty must be number type");

		using base = std::vector<Ty, Alloc>;
		// TODO
		NumVector() {
		}

		NumVector(std::initializer_list<Ty> init_list, const Alloc& al = Alloc()): base(init_list, al) {
		}

		NumVector(const NumVector& right): base(right) {
		}

		NumVector(const base& base_right): base(base_right) {
		}

		NumVector(const typename base::size_type& count, const Alloc& al = Alloc()): base(count, al) {
		}

		template <typename Iter>
		NumVector(Iter first, Iter last, const Alloc& al = Alloc()): base(first, last, al) {
		}

		template <typename Iter>
		NumVector(Iter first, const typename base::size_type& count, const Alloc& al = Alloc())
			: base(first, first + count, al) {
		}

		DEFINE_OPERATOR(+)
		DEFINE_OPERATOR(-)
		DEFINE_OPERATOR(*)
		DEFINE_OPERATOR(/)

		DEFINE_ASSIGN_OPERATOR(+=)
		DEFINE_ASSIGN_OPERATOR(-=)
		DEFINE_ASSIGN_OPERATOR(*=)
		DEFINE_ASSIGN_OPERATOR(/=)
	};

#undef DEFINE_OPERATOR
#undef DEFINE_ASSIGN_OPERATOR