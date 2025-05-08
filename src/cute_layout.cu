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

void constructing_layout() {
    printf("==========================================================\n");
    printf("                  coordinate_mapping \n");
    printf("==========================================================\n");
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

    print_layout(s2xh4);
    print_layout(s2xd4_a);
    print_layout(s2xh4_col);
    print_layout(s2xh4);
}

void coordinate_mapping() {
    printf("==========================================================\n");
    printf("                  coordinate_mapping \n");
    printf("==========================================================\n");
    // Coordinate Mapping  坐标映射
    auto shape_1 = Shape<_3, Shape<_2, _3>>{};
    /*
    - 以形状(3,(2,3)) 为例。此形状有三个坐标集：1-D 坐标、2-D 坐标和自然 （h-D） 坐标。

    1-D |     2-D     |    Natural           | 1-D |     2-D     |     Natural
    0   | (0,0) (0,0）| (0,(0,0)) (0,(0,0））|  9  | (0,3) (0,3）| (0,(1,1))（0,(1,1））
    1   | (1,0) (1,0）| (1,(0,0)) (1,(0,0））|  10 | (1,3) (1,3）| (1,(1,1))（1,(1,1））
    2   | (2,0) (2,0）| (2,(0,0)) (2,(0,0））|  11 | (2,3) (2,3）| (2,(1,1))（2,(1,1））
    3   | (0,1) (0,1）| (0,(1,0)) (0,(1,0））|  12 | (0,4) (0,4）| (0,(0,2))（0,(0,2））
    4   | (1,1) (1,1）| (1,(1,0)) (1,(1,0））|  13 | (1,4) (1,4）| (1,(0,2))（1,(0,2））
    5   | (2,1) (2,1）| (2,(1,0)) (2,(1,0））|  14 | (2,4) (2,4）| (2,(0,2))（2,(0,2））
    6   | (0,2) (0,2）| (0,(0,1)) (0,(0,1））|  15 | (0,5) (0,5）| (0,(1,2))（0,(1,2））
    7   | (1,2) (1,2）| (1,(0,1)) (1,(0,1））|  16 | (1,5) (1,5）| (1,(1,2))（1,(1,2））
    8   | (2,2) (2,2）| (2,(0,1)) (2,(0,1））|  17 | (2,5) (2,5）| (2,(1,2))（2,(1,2））
    */

    print(idx2crd(16, shape_1));                                    // (1, (1, 2))
    print(idx2crd(_16{}, shape_1));                                 // (_1, (_1, _2))
    print(idx2crd(make_coord(1, 5), shape_1));                      // (1, (1, 2))
    print(idx2crd(make_coord(_1{}, 5), shape_1));                   // (_1, (1, 2))
    print(idx2crd(make_coord(1, make_coord(1, 2)), shape_1));       // (1, (1, 2))
    print(idx2crd(make_coord(_1{}, make_coord(1, _2{})), shape_1)); // (_1, (1, _2))
    print("\n");
}

void index_mapping() {
    printf("==========================================================\n");
    printf("                  index_mapping \n");
    printf("==========================================================\n");
    // Index Mapping 索引映射
    auto shape_2 = Shape<_3, Shape<_2, _3>>{};
    auto stride_2 = Stride<_3, Stride<_12, _1>>{};
    /*
    从自然坐标到索引的映射是通过将自然坐标的内积与 Layout 的 Stride 来执行的。
    以布局(3,(2,3)): (3,(12,1)) 为例。那么自然坐标 (i,(j,k))
    将产生索引 i*3 + j*12 + k*1。此布局计算的索引显示在下面的 2-D 表中，其中i用作行坐标，(j,k)用作列坐标。

           0     1     2     3     4     5     <== 1-D col coord
         (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
        +-----+-----+-----+-----+-----+-----+
     0  |  0  |  12 |  1  |  13 |  2  |  14 |
        +-----+-----+-----+-----+-----+-----+
     1  |  3  |  15 |  4  |  16 |  5  |  17 |
        +-----+-----+-----+-----+-----+-----+
     2  |  6  |  18 |  7  |  19 |  8  |  20 |
        +-----+-----+-----+-----+-----+-----+
   */
    print(crd2idx(16, shape_2, stride_2));                                       // 17
    print(crd2idx(_16{}, shape_2, stride_2));                                    // _17
    print(crd2idx(make_coord(1, 5), shape_2, stride_2));                         // 17
    print(crd2idx(make_coord(_1{}, 5), shape_2, stride_2));                      // 17
    print(crd2idx(make_coord(_1{}, _5{}), shape_2, stride_2));                   // _17
    print(crd2idx(make_coord(1, make_coord(1, 2)), shape_2, stride_2));          // 17
    print(crd2idx(make_coord(_1{}, make_coord(_1{}, _2{})), shape_2, stride_2)); // _17
    print("\n");
}

void sublayouts() {
    printf("==========================================================\n");
    printf("                  sublayouts\n");
    printf("==========================================================\n");
    // Sublayouts 子布局
    Layout a = Layout<Shape<_4, Shape<_3, _6>>>{}; // (4,(3,6)) : (1,(4,12))
    Layout a0 = layout<0>(a);                      // 4:1
    Layout a1 = layout<1>(a);                      // (3,6):(4:12)
    Layout a10 = layout<1, 0>(a);                  // 3:4
    Layout a11 = layout<1, 1>(a);                  // 6:12

    PRINT_SHAPE(a);
    PRINT_SHAPE(a0);
    PRINT_SHAPE(a1);
    PRINT_SHAPE(a10);
    PRINT_SHAPE(a11);

    Layout b = Layout<Shape<_2, _3, _5, _7>>{}; // (2, 3, 5, 7):(1, 2, 6, 30)
    Layout b13 = select<1, 3>(b);               // (3, 7):(2, 30)
    Layout b01 = select<0, 1, 3>(b);            // (2, 3, 7):(1, 2, 30)
    Layout b2 = select<2>(b);                   // (5):(6)

    PRINT_SHAPE(b);
    PRINT_SHAPE(b13);
    PRINT_SHAPE(b01);
    PRINT_SHAPE(b2);

    Layout c = Layout<Shape<_2, _3, _5, _7>>{}; // (2, 3, 5, 7): (1, 2, 6, 30)
    Layout c13 = take<1, 3>(c);                 // (3, 5) : (2, 6)
    Layout c14 = take<1, 4>(c);                 // (3, 5, 7) : (2, 6, 30)

    PRINT_SHAPE(c);
    PRINT_SHAPE(c13);
    PRINT_SHAPE(c14);
}

void concatenation() {
    printf("==========================================================\n");
    printf("                  concatenation \n");
    printf("==========================================================\n");
    // Concatenation  串联
    Layout a = Layout<_3, _1>{};                  // 3:1
    Layout b = Layout<_4, _3>{};                  // 4:3
    Layout row = make_layout(a, b);               // (3, 4) : (1, 3)
    Layout col = make_layout(b, a);               // (4, 3) : (3, 1)
    Layout q = make_layout(row, col);             // ((3, 4), (4, 3)) : ((1, 3), (3, 1))
    Layout aa = make_layout(a);                   // (3) : (1)
    Layout aaa = make_layout(aa);                 // ((3)):((1))
    Layout d = make_layout(a, make_layout(a), a); // (3, (3), 3) : (1, (1), 1)

    PRINT_SHAPE(a);
    PRINT_SHAPE(b);
    PRINT_SHAPE(row);
    PRINT_SHAPE(col);
    PRINT_SHAPE(q);
    PRINT_SHAPE(aa);
    PRINT_SHAPE(aaa);
    PRINT_SHAPE(d);

    Layout ab = append(a, b);     // (3, 4):(1, 3)
    Layout ba = prepend(a, b);    // (4, 3):(3, 1)
    Layout c = append(ab, ba);    // (3, 4, (3, 4)) : (1, 3, (1, 3))
    Layout d1 = replace<2>(c, b); // (3, 4, 4) : (1, 3, 3)

    PRINT_SHAPE(ab);
    PRINT_SHAPE(ba);
    PRINT_SHAPE(c);
    PRINT_SHAPE(d1);
}

void grouping_and_flattening() {
    printf("==========================================================\n");
    printf("                  grouping_and_flattening \n");
    printf("==========================================================\n");
    Layout a = Layout<Shape<_2, _3, _5, _7>>{}; // (_2, _3, _5, _7):(_1, _2, _6, _30)
    Layout b = group<0, 2>(a);                  // ((_2, _3), _5, _7):((_1, _2), _6, _30)
    Layout c = group<1, 3>(a);                  // ((_2, _3), (_5, _7)):((_1, _2), (_6, _30))
    Layout f = flatten(b);                      // (_2, _3, _5, _7):(_1, _2, _6, _30)
    Layout e = flatten(c);                      // (_2, _3, _5, _7):(_1, _2, _6, _30)

    PRINT_SHAPE(a);
    PRINT_SHAPE(b);
    PRINT_SHAPE(c);
    PRINT_SHAPE(f);
    PRINT_SHAPE(e);
}

int main(int argc, char **argv) {
    constructing_layout();     // 构建布局
    coordinate_mapping();      // 坐标映射
    index_mapping();           // 索引映射
    sublayouts();              // 子布局
    concatenation();           // 串联
    grouping_and_flattening(); // 分组和展开

    return 0;
}
