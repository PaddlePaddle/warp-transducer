#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif
