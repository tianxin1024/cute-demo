#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
    Layout ThrId = make_layout(make_shape(4, 2), make_stride(1, 16));
    print_layout(ThrId);

    Layout CLayout = make_layout(make_shape(8, 8));
    print_layout(CLayout);

    Layout CLayout_1 = make_layout(make_shape(make_shape(2, 2, 2), 8), make_stride(make_stride(1, 16, 4), 4));
    print_layout(CLayout_1);

    Layout CLayout_2 = make_layout(make_shape(make_shape(2, 2, 2), make_shape(2, 2, 2)),
                                   make_stride(make_stride(1, 16, 4), make_stride(8, 2, 32)));
    print_layout(CLayout_2);

    Layout ALayout = make_layout(make_shape(make_shape(4, 2), 4),
                                 make_stride(make_stride(8, 4), 1));
    print_layout(ALayout);
    return 0;
}
