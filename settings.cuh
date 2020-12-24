#ifndef SETTINGS_CUH
#define SETTINGS_CUH

#include <string>
#include <vector>

#include "utils.cuh"


struct Settings {
    TubeType tube;
    float voltage, power;
    float *energy;
    float *spectrum;
    float flux;
    float exposure;

    __host__ void init(TubeType tube_, float voltage_, float power_, float exposure_) {
        tube = tube_;
        voltage = voltage_;
        power = power_;
        exposure = exposure_;

        std::string tube_str;

        switch(tube) {
            case Be_08:
                tube_str = "Be 0.8 mm";
                break;
            case Be_30:
                tube_str = "Be 3.0 mm";
                break;
            case Be_50:
                tube_str = "Be 5.0 mm";
                break;
        }
        
        // D:/Proj_One/GitSeparate/xray_cuda
        std::string root_path = std::string("./data/tubes/X-Ray W11D ") + tube_str + "/";
        std::string flux_path = root_path + tube_str + " Fg 100 cm 1 kW.txt";
        std::string spectrum_path = root_path + "W11D " + std::to_string(int(voltage)) + " kV " + tube_str + ".txt";
        
        std::vector<float> energy_vec, spectrum_vec;
        read_data(spectrum_path, energy_vec, spectrum_vec);

        std::vector<float> energy_vec_fl, flux_vec;
        read_data(flux_path, energy_vec_fl, flux_vec);

        cudaMallocManaged(&energy, energy_vec.size()*sizeof(float));
        cudaMallocManaged(&spectrum, energy_vec.size()*sizeof(float));

        copy(energy_vec.begin(), energy_vec.end(), energy);
        copy(spectrum_vec.begin(), spectrum_vec.end(), spectrum);

        flux = -1.0;
        for(int i = 0; i < energy_vec_fl.size(); i++) {
            if(abs(voltage - energy_vec_fl[i]) < FLT_EPS) {
                flux = flux_vec[i] * power;
                break;
            }
        }
        if(abs(flux + 1.0) < FLT_EPS) {
            throw "There is no appropriate value of flux for the given voltage.";
        }
    }
};

#endif