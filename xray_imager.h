#pragma once

#ifndef KERNEL_H
#define KERNEL_H

#include "types.h"

#ifdef CUDA_XRAY_DLL_LIB
    #define CUDA_XRAY_DLL_API __declspec(dllexport)
#else
    #define CUDA_XRAY_DLL_API __declspec(dllimport)
#endif

extern "C" CUDA_XRAY_DLL_API float** xray_image(
    TubeType tube_type,
    float volatage,
    float power,
    float det_resolution,
    float det_size,
    float det_exposure,
    float hole_size,
    float p1_thicc,
    float p2_thicc
);

#endif  // KERNEL_H