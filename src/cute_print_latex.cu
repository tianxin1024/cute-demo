#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
#if 0
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
#endif

#if 0
    Layout ALayout_A = make_layout(make_shape(make_shape(4, 8), 1),
                                   make_stride(make_stride(8, 1), 0));
    print_layout(ALayout_A);

    Layout ALayout_B = make_layout(make_shape(make_shape(4, 8), 2),
                                   make_stride(make_stride(16, 1), 8));
    print_layout(ALayout_B);
#endif

#if 0
    TiledMMA mmaC_v0 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{});
    print_latex(mmaC_v0);
#endif

#if 0
    TiledMMA mmaC_v1 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_2, _2, _1>>{}, // Thr layout
        Layout<Shape<_1, _1, _1>>{}  // Val layout
    );
    print_latex(mmaC_v1);
#endif

#if 0
    TiledMMA mmaC_v2 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_2, _1, _1>>{}, // Thr layout
        Layout<Shape<_1, _1, _1>>{}  // Val layout
    );
    print_latex(mmaC_v2);
#endif

#if 0
    TiledMMA mmaC_v3 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_1, _1, _2>>{}, // Thr layout
        Layout<Shape<_1, _1, _1>>{}  // Val layout
    );
    print_latex(mmaC_v3);
#endif

#if 0
    TiledMMA mmaC_v4 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_1, _2, _1>>{}, // Thr layout
        Layout<Shape<_1, _1, _1>>{}  // Val layout
    );
    print_latex(mmaC_v4);
#endif

#if 0
    TiledMMA mmaC_v5 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_1, _1, _1>>{}, // Thr layout
        Layout<Shape<_2, _2, _1>>{}  // Val layout
    );
    print_latex(mmaC_v5);
#endif

#if 0
    TiledMMA mmaC_v6 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_8, _8, Layout<Shape<_4, _2>, Stride<_2, _1>>>{});
    print_latex(mmaC_v6);

#endif

#if 0
    TiledMMA mmaC_v7 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_1, _1, _1>>{},
        Tile<_8, _8, Layout<Shape<_2, _4>, Stride<_4, _1>>>{});
    print_latex(mmaC_v7);
#endif

#if 0
    TiledMMA mmaC_v8 = make_tiled_mma(
        SM80_8x8x4_F64F64F64F64_TN{},
        Layout<Shape<_2, _1, _1>>{},
        // Layout<Shape<_1, _1, _1>>{},
        Tile<Layout<_1, _1>, Layout<_1, _1>, Layout<Shape<_2, _4>, Stride<_4, _1>>>{});
    print_latex(mmaC_v8);
#endif

#if 1
    TiledMMA mmaC = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
                                   Layout<Shape<_1, _1>>{});
    Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t> s2r_atom_A;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mmaC);
    print_latex(s2r_copy_a);
#endif

    return 0;
}
