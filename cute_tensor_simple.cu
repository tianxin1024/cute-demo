#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

int main() {
    srand(10086);

    /*
    由于cutlass/cute的设计目标是极致的高性能计算，所以cute需要编译器时期确定静态形状，不支持动态分配状态。
    */

#if 1
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
#endif

    return 0;
}
