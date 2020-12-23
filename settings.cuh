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
    float *flux;

    __host__ void init(TubeType tube_, float voltage_, float power_) {
        tube = tube_;
        voltage = voltage_;
        power = power_;

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
        
        std::string root_path = std::string("./data/tubes/X-Ray W11D ") + tube_str + "/";
        std::string flux_path = root_path + tube_str + " Fg 100 cm 1 kW.txt";
        std::string spectrum_path = root_path + "W11D " + std::to_string(int(voltage)) + " kV " + tube_str + ".txt";
        
        std::vector<float> energy_vec, spectrum_vec;
        read_data(spectrum_path, energy_vec, spectrum_vec);

        std::vector<float> energy_vec_fl, flux_vec;
        read_data(flux_path, energy_vec_fl, flux_vec);

        cudaMallocManaged(&energy, energy_vec.size()*sizeof(float));
        cudaMallocManaged(&spectrum, energy_vec.size()*sizeof(float));
        cudaMallocManaged(&flux, energy_vec.size()*sizeof(float));

        int k = 0;
        for(int i = 0; i < energy_vec.size(); i++) {
            // copy energy and spectrun
            energy[i] = energy_vec[i];
            spectrum[i] = spectrum_vec[i];

            // interpolate flux
            while(energy[i] >= energy_vec_fl[k]) {
                k++;
            }
            float flux_k = (flux_vec[k] - flux_vec[k-1]) / (energy_vec_fl[k] - energy_vec_fl[k-1]);
            // also scale it to used power
            flux[i] = power * (flux_vec[k-1] + (energy[i] - energy_vec_fl[k-1]) * flux_k);
        }
    }
};

#endif