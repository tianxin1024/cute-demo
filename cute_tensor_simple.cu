#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

template <class Tensor>
void print_information(Tensor const &tensor) {
    auto layout = cute::layout<>(tensor);
    printf("\tlayout : ");
    print(layout);
    printf("\n");

    auto shape = cute::shape<>(tensor);
    printf("\tshape  : ");
    print(shape);
    printf("\n");

    auto stride = cute::stride<>(tensor);
    printf("\tstride : ");
    print(stride);
    printf("\n");

    auto size = cute::size<>(tensor);
    printf("\tsize   : ");
    print(size);
    printf("\n");

    auto rank = cute::rank<>(tensor);
    printf("\trank   : ");
    print(rank);
    printf("\n");

    auto depth = cute::depth<>(tensor);
    printf("\tdepth  : ");
    print(depth);
    printf("\n");
}

void create_tensor() {
    // =========================================================================
    //    1. 基本Tensor创建
    // =========================================================================

    // 静态形状Tensor
    constexpr int M = 4;
    constexpr int N = 6;
    constexpr int K = 5;

    // 创建一个2D Tensor (4x6)
    Tensor tensor_a_2d = make_tensor<float>(make_shape(Int<M>{}, Int<N>{}));
    Tensor tensor_b_2d = make_tensor<float>(Shape<Int<M>, Int<N>>{});

    Tensor tensor_1 = make_tensor<float>(Shape<_4, _6>{}); // 也可以直接用_表示
    print_tensor(tensor_1);
    // print(tensor_a_2d);  // print会打印tensor存储空间所在的位置和shape、stride信息
    print_tensor(tensor_a_2d); // print_tensor还会打印数值信息
    printf("\n");
    // print(tensor_b_2d);
    print_tensor(tensor_b_2d);
    printf("\n");

    // 创建一个3D Tensor (4x6x5) 并初始化
    Tensor tensor_3d = make_tensor<float>(make_shape(Int<M>{}, Int<N>{}, Int<K>{}));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                tensor_3d(m, n, k) = m * 100 + n * 10 + k;
            }
        }
    }
    // print(tensor_3d);
    print_tensor(tensor_3d);
    printf("\n");
}

void tensor_base() {
    using T = cute::half_t;

    T *Aptr_host;
    T *Bptr_host;

    int m = 4;
    int n = 6;
    int k = 5;

    Aptr_host = (T *)malloc(sizeof(T) * m * n);
    Bptr_host = (T *)malloc(sizeof(T) * m * n);

    // fill number column major
    for (int i = 0; i < m * n; ++i) {
        Aptr_host[i] = i;
        Bptr_host[i] = i;
    }

    // tensor 创建
    Tensor tensor_A = make_tensor(Aptr_host, make_shape(m, n), make_stride(n, 1)); // 行优先 row-major
    Tensor tensor_B = make_tensor(Bptr_host, make_shape(n, m), make_stride(1, n)); // 列优先 column-major
    printf("======================== Tensor A ===============================\n");
    print_tensor(tensor_A);
    print_information<>(tensor_A);

    // 维度信息查询
    printf("======================== Tensor B ===============================\n");
    print_tensor(tensor_B);
    print_information(tensor_B);

    // 修改访问信息，operator()/operator[]
    printf("======================== Tensor operator ==========================\n");
    tensor_A(0) = 100;
    tensor_A(1, 2) = 101;
    auto coord = make_coord(2, 3);
    tensor_A(coord) = 102;
    print_tensor(tensor_A);

    T *Cptr_host;
    Cptr_host = (T *)malloc(sizeof(T) * m * n * k);

    for (int i = 0; i < m * n * k; ++i) {
        Cptr_host[i] = i;
    }
    Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n, k)); // MxNxK

    // slice 筛选特定轴
    printf("======================== Tensor C ===============================\n");
    print_tensor(tensor_C);
    printf("======================== Tensor slice k ===========================\n");
    Tensor tensor_slice_k = tensor_C(_, _, 3); // MxN,k=3
    print_tensor(tensor_slice_k);
    printf("======================== Tensor slice m ===========================\n");
    Tensor tensor_slice_m = tensor_C(3, _, _);
    print_tensor(tensor_slice_m);

    printf("======================== Tensor Take  =============================\n");
    // 通过take函数提取[B, E)的轴上数据
    Tensor tensor_take = take<2, 3>(tensor_C);
    print_tensor(tensor_take);

    printf("======================== Tensor flatten =============================\n");
    Tensor tensor_flatten = flatten(tensor_C); // M, N, K
    print_tensor(tensor_flatten);
}

void layout_demo() {
    Layout layout_row = make_layout(make_shape(_4{}, _6{}),
                                    make_stride(_6{}, _1{}));
    print_layout(layout_row);

    Layout layout_col = make_layout(make_shape(_4{}, _6{}),
                                    make_shape(_1{}, _4{}));
    print_layout(layout_col);
}

void tensor_tile() {
    using T = cute::half_t;

    T *Aptr_host;
    T *Bptr_host;

    int m = 4;
    int n = 6;

    Aptr_host = (T *)malloc(sizeof(T) * m * n);
    Bptr_host = (T *)malloc(sizeof(T) * m * n);

    // fill number column major
    for (int i = 0; i < m * n; ++i) {
        Aptr_host[i] = i;
        Bptr_host[i] = i;
    }

    auto row_coord = make_coord(1, 3);
    auto col_coord = make_coord(2, 4);
    auto coord = make_coord(row_coord, col_coord);
    print(coord);
    printf("\n");
}

int main() {
    srand(10086);

    /*


    */

    /*
     *  由于cutlass/cute的设计目标是极致的高性能计算，所以cute需要编译器时期确定静态形状，要注意动态分配状态。
     *  Tensor 的创建
     *       1) 栈上对象：需同时指定类型和Layout, layout必须是静态shape
     *          Tensor make_tensor<T>(Layout layout);
     *       2) 堆上对象：需指定pointer和Layout， layout可动可静
     *          Tensor make_tensor(Pointer pointer, Layout layout);
     *       3) 栈上对象：tensor的layout必须是静态的
     *          Tensor make_tensor_like(Tensor tensor);
     *       4) 堆上对象：tensor的layout必须是静态的
     *          Tensor make_fragment_like(Tensor tensor);
     */
    // create_tensor();

    // tensor_base();

    layout_demo();

    tensor_tile();

    return 0;
}
