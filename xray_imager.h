/**
 * The CUDA x-ray demo dll which calculates an x-ray image.
 *
 * @author  Valeriy Lyubich
 * @version 0.1
 * @since   2020-12-29
 */

#pragma once

#ifndef KERNEL_H
#define KERNEL_H


#include "./utils/utils_host.h"
#include "./utils/types.h"
#include "./core/settings.h"


#ifdef CUDA_XRAY_DLL_LIB
    #define CUDA_XRAY_DLL_API __declspec(dllexport)
#else
    #define CUDA_XRAY_DLL_API __declspec(dllimport)
#endif

/**
 * X-ray imager entry point, it is used for x-ray image calculation
 * @param settings x-ray settings @see Settings
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
    Settings settings,
    PartType part_type,
    float hole_size,
    float p1_thicc
);

// extern "C" CUDA_XRAY_DLL_API void free_image();

#endif  // KERNEL_H