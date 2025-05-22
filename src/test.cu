#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
#if 0
    // Layout a = make_layout(make_shape(8, 4));
    // print_layout(a);

    // Layout b = make_layout(make_shape(8, 4),
    //                        make_stride(4, 1));
    // print_layout(b);

    // Layout ThrId = make_layout(make_shape(4, 2), make_stride(1, 16));
    // print_layout(ThrId);

    Layout a = make_layout(make_shape(4, 8));
    print_layout(a);

    Layout b = make_layout(make_shape(4, 8), make_stride(8, 1));
    print_layout(b);

    auto shape_b = Shape<_4, _8>{};
    printf("输出shape_b(4, 8)在第一行的自然坐标\n");
    print(idx2crd(0, shape_b));
    print(idx2crd(1, shape_b));
    print(idx2crd(2, shape_b));
    print(idx2crd(3, shape_b));
    print(idx2crd(4, shape_b));
    print(idx2crd(5, shape_b));
    print(idx2crd(6, shape_b));
    print(idx2crd(7, shape_b));
    print("\n");

    Layout c_right = make_layout(make_shape(4, make_shape(2, 4)), LayoutRight{});
    print_layout(c_right);

    Layout c_left = make_layout(make_shape(4, make_shape(2, 4)), LayoutLeft{});
    print_layout(c_left);

    Layout c = make_layout(make_shape(4, make_shape(2, 4)),
                           make_stride(1, make_stride(4, 8)));
    print_layout(c);

    auto shape_c = Shape<_4, Shape<_2, _4>>{};
    auto stride_c = Stride<_1, Stride<_4, _8>>{};
    printf("输出shape_c(4, (2, 4))在第一行的自然坐标\n");
    for (int i = 0; i < 4; i++) {
        print("\n");
        for (int j = 0; j < 8; j++) {
            print(idx2crd(j * 4 + i, shape_c));
            printf(" ");
        }
    }
    print("\n");

    print(crd2idx(make_coord(0, make_coord(1, 0)), shape_c, stride_c));
    print("\n");

    Layout d = make_layout(make_shape(4, make_shape(2, 4)),
                           make_stride(2, make_stride(1, 8)));
    print_layout(d);

    auto shape_d = Shape<_4, Shape<_2, _4>>{};
    auto stride_d = Stride<_2, Stride<_1, _8>>{};
    printf("输出shape_d(4, (2, 4))在第一行的自然坐标\n");
    for (int i = 0; i < 4; i++) {
        print("\n");
        for (int j = 0; j < 8; j++) {
            print(idx2crd(j * 4 + i, shape_d));
            printf(" ");
        }
    }
    print("\n");

    print(crd2idx(make_coord(0, make_coord(1, 0)), shape_d, stride_d));
    print("\n");

    Layout e = make_layout(make_shape(make_shape(2, 2), make_shape(2, 4)),
                           LayoutLeft{});
    print_layout(e);

    auto shape_e = Shape<Shape<_2, _2>, Shape<_2, _4>>{};
    printf("输出shape_c((2, 2), (2, 4))在第一行的自然坐标\n");
    for (int i = 0; i < 4; i++) {
        print("\n");
        for (int j = 0; j < 8; j++) {
            print(idx2crd(j * 4 + i, shape_e));
            printf(" ");
        }
    }
    print("\n");

    Layout f = make_layout(make_shape(make_shape(2, 2), make_shape(2, 4)),
                           make_stride(make_stride(1, 4), make_stride(2, 8)));
    print_layout(f);

    /* 哈哈，我终于找见规律了，就是先将shape打印出2维的自然坐标，再与原来的坐标进行对比。
    举例说明：
    假如我现在要生成一个((2,2)(2,4)):((1,4)(2,8))的layout，如图所示
     ((2,2),(2,4)):((1,4),(2,8))
           0    1    2    3    4    5    6    7
        +----+----+----+----+----+----+----+----+
     0  |  0 |  2 |  8 | 10 | 16 | 18 | 24 | 26 |
        +----+----+----+----+----+----+----+----+
     1  |  1 |  3 |  9 | 11 | 17 | 19 | 25 | 27 |
        +----+----+----+----+----+----+----+----+
     2  |  4 |  6 | 12 | 14 | 20 | 22 | 28 | 30 |
        +----+----+----+----+----+----+----+----+
     3  |  5 |  7 | 13 | 15 | 21 | 23 | 29 | 31 |
        +----+----+----+----+----+----+----+----+
    我先将(4, 8)的自然坐标进行打印
    ((0,0),(0,0)) ((0,0),(1,0)) ((0,0),(0,1)) ((0,0),(1,1)) ((0,0),(0,2)) ((0,0),(1,2)) ((0,0),(0,3)) ((0,0),(1,3))
    ((1,0),(0,0)) ((1,0),(1,0)) ((1,0),(0,1)) ((1,0),(1,1)) ((1,0),(0,2)) ((1,0),(1,2)) ((1,0),(0,3)) ((1,0),(1,3))
    ((0,1),(0,0)) ((0,1),(1,0)) ((0,1),(0,1)) ((0,1),(1,1)) ((0,1),(0,2)) ((0,1),(1,2)) ((0,1),(0,3)) ((0,1),(1,3))
    ((1,1),(0,0)) ((1,1),(1,0)) ((1,1),(0,1)) ((1,1),(1,1)) ((1,1),(0,2)) ((1,1),(1,2)) ((1,1),(0,3)) ((1,1),(1,3))

    假如现在不知道stride是多少，先设((x1, y1), (x2, y2))
    可以知道：
    1 * x1 + 0 * y1 + 0 * x2 + 0 * y2 = 1     >>  x1 = 1
    0 * x1 + 1 * y1 + 0 * x2 + 0 * y2 = 4     >>  y1 = 4
    0 * x1 + 0 * y1 + 1 * x2 + 0 * y2 = 2     >>  x2 = 2
    0 * x1 + 0 * y1 + 0 * x2 + 1 * y2 = 8     >>  y2 = 8

    所以stride是((1, 4), (2, 8))
    */

#endif

#if 0
    TiledMMA mmaC_v1 = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<_2, _2, _1>>{}, // Thr layout
        Layout<Shape<_1, _2, _1>>{}  // Val layout
    );
    print(mmaC_v1);
#endif

#if 1
    // Tile 仅定义形状，不涉及布局的步长或排列。
    // Layout 不仅定义形状，还包含步长和排列信息（即使未显式指定，也可能有默认值）。
    TiledMMA mma = make_tiled_mma(
        SM80_16x8x8_F16F16F16F16_TN{},
        Layout<Shape<_2, _2>>{},
        Tile<_32, _32, _16>{});
    // print(mma);
    // print_latex(mma);

    TiledMMA mma_1 = make_tiled_mma(
        SM80_16x8x8_F16F16F16F16_TN{},
        Layout<Shape<_2, _2>>{},
        Layout<Shape<_32, _32, _16>>{});
    // print(mma_1);
    // print_latex(mma_1);

    TiledMMA mma_2 = make_tiled_mma(
        SM80_16x8x8_F16F16F16F16_TN{},
        Layout<Shape<_2, _2, _1>>{},
        Layout<Shape<_16, _1, _2>>{});
    print(mma_2);
    // print(mma_2);
    // print_latex(mma_2);
#endif

    return 0;
}
