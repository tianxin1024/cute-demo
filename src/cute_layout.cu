// link: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md
#include <cute/tensor.hpp>
#include <iostream>
#include <string>

using namespace cute;

#define PRINT_SHAPE(var) print_shape(var, #var)
#define PRINT2D(var) print2D(var, #var)
#define PRINT1D(var) print1D(var, #var)

template <class Layout>
void print_shape(Layout const &lay, std::string const &name) {
    printf("[%-9s] : ", name.c_str());
    print(lay);
    printf("\n");
}

template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const &layout, std::string const &name) {
    printf(">>> print2D(%s):  \n-------------------------\n", name.c_str());
    for (int m = 0; m < size<0>(layout); ++m) {
        for (int n = 0; n < size<1>(layout); ++n) {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
}

template <class Shape, class Stride>
void print1D(Layout<Shape, Stride> const &layout, std::string const &name) {
    printf(">>> print1D(%s):  \n-------------------------\n", name.c_str());
    for (int i = 0; i < size(layout); ++i) {
        printf("%3d  ", layout(i));
    }
    printf("\n");
}

int main(int argc, char **argv) {
    Layout s8 = make_layout(Int<8>{});
    Layout d8 = make_layout(8);

    // 如果没有stride，默认为LayoutLeft, 也就是col-major
    Layout s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{}, 4));

    Layout s2xd4_a = make_layout(make_shape(Int<2>{}, 4), make_stride(Int<12>{}, Int<1>{}));
    Layout s2xd4_col = make_layout(make_shape(Int<2>{}, 4), LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{}, 4), LayoutRight{});

    Layout s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                               make_stride(4, make_stride(2, 1)));
    Layout s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});
    PRINT_SHAPE(d8);
    PRINT_SHAPE(s8);
    PRINT_SHAPE(s2xs4);
    PRINT_SHAPE(s2xd4);
    PRINT_SHAPE(s2xd4_a);
    PRINT_SHAPE(s2xd4_col);
    PRINT_SHAPE(s2xd4_row);
    PRINT_SHAPE(s2xh4);
    PRINT_SHAPE(s2xh4_col);

    PRINT2D(s2xs4);
    PRINT2D(s2xd4_a);
    PRINT2D(s2xh4_col);
    PRINT2D(s2xh4);

    PRINT1D(s2xs4);
    PRINT1D(s2xd4_a);
    PRINT1D(s2xh4_col);
    PRINT1D(s2xh4);

    return 0;
}
