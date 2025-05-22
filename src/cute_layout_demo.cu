#include <cute/tensor.hpp>

using namespace cute;

int main() {
    Layout a0 = make_layout(make_shape(4, make_shape(2, 4)));
    print_layout(a0);

    Layout a = make_layout(make_shape(4, make_shape(2, 4)),
                           make_stride(2, make_stride(1, 8)));
    print_layout(a);

    Layout b = make_layout(make_shape(make_shape(4, 1), make_shape(2, 4)),
                           make_stride(make_stride(2, 1), make_stride(1, 8)));
    print_layout(b);
    return 0;
}
