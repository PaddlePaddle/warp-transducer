#pragma once
#include "rnnt_helper.h"
#include "rnnt.h"

const int warp_size = 32;

template<int NT, typename T, typename Rop>
struct CTAReduce;

template<int NT, typename T, typename Rop>
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    __device__ static T reduce(int tid, T x, Storage& storage, int count, Rop g) {
        T* s = storage.shared;
        s[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass. gather from shared mem into warp.
#pragma unroll
        for(int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if(tid + offset < count && tid < offset) {
                // Read from the right half and store to the left half.
                x = g(x, s[offset + tid]);
                s[tid] = x;
            }
            __syncthreads();
        }

        // gather in warp.
        T shuff;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#if CUDART_VERSION < 9000
            shuff = __shfl_down(x, offset);
#else
            shuff = __shfl_down_sync(0xFFFFFFFF, x, offset);
#endif
            if (tid + offset < count && tid < offset)
                x = g(x, shuff);
        }
        return x;
    }
};

// num_rows = len of row = cols, num_cols = len of col = rows
// reduce_max(acts, denom, num_rows=alphabet_size_, num_cols=minibatch_ * maxT_ * maxU_, 0 stream_);
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T* const acts, T* output, int num_rows) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage; // storage size = NT = blockDim.x

    int tid = threadIdx.x; // blockDim.x = NT, work on axis which size is `alphabet_size_`
    int idx = tid; // 0 <= tid < NT
    int col = blockIdx.x; //  gridDim.x = minibatch_ * maxT_ * maxU_, col=row_idx
    T curr;

    // Each block works on a column, gather idx val with  NT stride
    if (idx < num_rows) {
        // f=identity, num_rows = cols
        curr = f(acts[col * num_rows + idx]);
    }
    idx += NT;

    while (idx < num_rows) {
        // g=max
        curr = g(curr, f(acts[col * num_rows + idx]));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0)
        output[col] = curr;
}

// reduce_exp(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 1, stream_);
template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_minus(Iop f, Rop g, const T* const acts, T* output, int num_rows) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;
    T max = output[col];

    // Each block works on a column
    if (idx < num_rows) {
        // f = exp
        curr = f(acts[col * num_rows + idx] - max);
    }
    idx += NT;

    while (idx < num_rows) {
        // g = sum
        curr = g(curr, f(acts[col * num_rows + idx] - max));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0)
        output[col] = -max - log(curr);
}

struct ReduceHelper {

    template<typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T* const acts, T* output, int num_rows, int num_cols, bool minus, cudaStream_t stream) {

        int grid_size;

        if (minus) {
            grid_size = num_cols;
            reduce_minus<128><<<grid_size, 128, 0, stream>>>
               (f, g, acts, output, num_rows);

        } else {
            grid_size = num_cols;
            reduce_rows<128><<<grid_size, 128, 0, stream>>>
               (f, g, acts, output, num_rows);
        }
    }
};


template<typename T, typename Iof, typename  Rof>
rnntStatus_t reduce(Iof f, Rof g, const T* const acts, T* output, int rows, int cols, bool minus, cudaStream_t stream) {
    ReduceHelper::impl(f, g, acts, output, rows, cols, minus, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return RNNT_STATUS_EXECUTION_FAILED;

    return RNNT_STATUS_SUCCESS;
}

// reduce_max(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 0, stream_);
// reduce_exp(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 1, stream_);
template<typename T>
rnntStatus_t reduce_exp(const T* const acts, T *denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::exponential<T>(), rnnt_helper::add<T>(), acts, denom, rows, cols, minus, stream);
}

template<typename T>
rnntStatus_t reduce_max(const T* const acts, T *denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::identity<T>(), rnnt_helper::maximum<T>(), acts, denom, rows, cols, minus, stream);
}

