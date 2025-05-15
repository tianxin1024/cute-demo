#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include "tools/gpu_clock.hpp"

#define PRINT 0
#define LATEX 0

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler, class TA, class AStride,
          class ASmemLayout, class TiledCopyA, class S2RAtomA, class TB,
          class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(
    TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK,
                                          CtaTiler cta_tiler, TA const *A,
                                          AStride dA, ASmemLayout sA_layout,
                                          TiledCopyA copy_a,
                                          S2RAtomA s2r_atom_a, TB const *B,
                                          BStride dB, BSmemLayout sB_layout,
                                          TiledCopyB copy_b,
                                          S2RAtomB s2r_atom_b, TC *C,
                                          CStride dC, CSmemLayout, TiledMma mma,
                                          Alpha alpha, Beta beta) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma)); // Numthreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<ASmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler)); // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler)); // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler)); // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler)); // BLK_K

    CUTE_STATIC_ASSERT_V(
        congruent(select<0, 2>(shape_MNK), dA)); // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(
        congruent(select<1, 2>(shape_MNK), dB)); // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(
        congruent(select<0, 1>(shape_MNK), dC)); // dC strides for shape MN

    // Full and Tiled Tensors
    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M, N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m, n, k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout); // (BLK_N, BLK_K, PIPE)

    // Partition the copying of A and B tiles across the threads
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K

    // PREFETCH

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) {
            ++k_tile_next;
        }
    }

    // Define A/B partitioning and C accumulators
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)

    CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0, 3>(shape(tCgC)))); // (MMA, MMA_M, MMA_N)
    CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));         // MMA_M
    CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));         // MMA_N

    // Clear the accumulators
    clear(tCrC);

    // Copy Atom retiling
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA); // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB); // (CPY, MMA_N, MMA_K, PIPE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, MMA_N, MMA_K)
}

template <class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, cute::half_t const *A, int ldA,
             cute::half_t const *B, int ldB, Beta beta, cute::half_t *C,
             int ldC, cudaStream_t stream = 0) {
    assert(false && "Not implemented");
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const *A, int ldA,
             TB const *B, int ldB, Beta beta, TC *C, int ldC,
             cudaStream_t stream = 0) {
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const *A, int ldA,
             TB const *B, int ldB, Beta beta, TC *C, int ldC,
             cudaStream_t stream = 0) {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<3>{};

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK, bP)); // (m, k, p) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK, bP)); // (n, k, p) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));     // (m, n) -> smem_idx; m-major

    // Define the thread layouts (static)
    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
                                      Layout<Shape<_4, _1>>{}); // Val layout 4x1 m-major

    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                                      Layout<Shape<_4, _1>>{}); // Val layout 4x1 n-major

    TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                   Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

#if PRINT
    printf("---------------------\n");
    print(copyA);
    printf("---------------------\n");
    print(copyB);
    printf("---------------------\n");
    print(mmaC);
#endif

#if LATEX
    // print_latex(copyA);
    // print_latex(copyB);
    print_latex(mmaC);
#endif

    int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(mmaC));                                  // (16, 16, 1)
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN))); //  (40, 40, 1)

    gemm_device<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, copyA, AutoVectorizingCopy{},
        B, dB, sB, copyB, AutoVectorizingCopy{},
        C, dC, sC, mmaC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha,
          TA const *A, int ldA, TB const *B, int ldB, Beta beta, TC *C, int ldC,
          cudaStream_t stream = 0) {
    if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } else if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not implemented");
}

int main(int argc, char **argv) {
#if 0
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    if (props.major < 8) {
        std::cout << "This example requires an Ampere GPU or newer (CC >= 80)" << std::endl;
        return 0;
    }

    std::cout << "Using device 0: " << props.name
              << " (SM" << props.major * 10 + props.minor
              << ", " << props.multiProcessorCount
              << ")" << std::endl;

    int m = 5120;
    if (argc >= 2)
        sscanf(argv[1], "%d", &m);

    int n = 5120;
    if (argc >= 3)
        sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4)
        sscanf(argv[3], "%d", &k);

    char transA = 'N';
    if (argc >= 5)
        sscanf(argv[4], "%c", &transA);

    char transB = 'T';
    if (argc >= 6)
        sscanf(argv[5], "%c", &transB);

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;
    std::cout << "C = A^" << transA << " B^" << transB << std::endl;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    for (int j = 0; j < m * k; ++j) {
        h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int j = 0; j < k * n; ++j) {
        h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int j = 0; j < m * n; ++j) {
        h_C[j] = static_cast<TC>(-1);
    }

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = (2.0 * m * n * k) * 1e-9;

    const int timing_iterations = 100;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    } else if (transB == 'T') {
        ldB = n;
    } else {
        assert(false);
    }

    // Run once
    d_C = h_C;
    gemm(transA, transB, m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(),
         ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> cute_result = d_C;

#endif
    return 0;
}
