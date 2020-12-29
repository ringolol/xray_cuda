#ifndef SETTINGS_CUH
#define SETTINGS_CUH

#include <string>
#include <vector>

#include "../utils/types.h"
#include "../utils/utils_host.h"


/**
 * X-ray imager setting
 */
struct Settings {
    TubeType tube;
    float voltage, power;
    float exposure;
    float det_resolution;
    float det_size;


    /**
     * Set settings
     * @param tube_
     * @param voltage_
     * @param power_
     * @param det_resolution_
     * @param det_size_
     * @param exposure_ 
     * 
     * @throw std::invalid_argument
     * @throw std::ios_base::failure
     */
    Settings(TubeType tube_, float voltage_, float power_, float det_resolution_, float det_size_, float exposure_) {
        init(tube_, voltage_, power_, det_resolution_, det_size_, exposure_);
    }

    void init(TubeType tube_, float voltage_, float power_, float det_resolution_, float det_size_, float exposure_) {
        tube = tube_;
        voltage = voltage_;
        power = power_;
        det_resolution = det_resolution_;
        det_size = det_size_;
        exposure = exposure_;
    }
};

#endif  // SETTINGS_CUH