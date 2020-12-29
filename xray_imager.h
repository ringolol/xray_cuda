#pragma once

#ifndef KERNEL_H
#define KERNEL_H

#include "types.h"

#ifdef CUDA_XRAY_DLL_LIB
    #define CUDA_XRAY_DLL_API __declspec(dllexport)
#else
    #define CUDA_XRAY_DLL_API __declspec(dllimport)
#endif

/**
 * X-ray imager entry point, it is used for x-ray image calculation
 * @param tube_type x-ray source tube type
 * @param voltage x-ray accelerating voltage [kV]
 * @param power x-ray source power [kW]
 * @param det_resolution detector resolution [px]
 * @param det_size detector phisical size [cm]
 * @param det_exposure detector exposure [s]
 * @param part_type part type (either bubble or notch)
 * @param hole_size part hole (or notch) size [cm]
 * @param p_thicc part plate thickness [cm]
 * 
 * @return x-ray image (number of photons which hit each pixel of detector matrix)
 * 
 * @throw std::invalid_argument
 * @throw std::ios_base::failure
 **/
extern "C" CUDA_XRAY_DLL_API float** xray_image(
    TubeType tube_type,
    float voltage,
    float power,
    float det_resolution,
    float det_size,
    float det_exposure,
    PartType part_type,
    float hole_size,
    float p1_thicc
);

#endif  // KERNEL_H