#ifndef SETTINGS_DEVICE_CUH
#define SETTINGS_DEVICE_CUH


#include "settings.h"


struct SettingsDevice : Settings {
    int energy_size;
    float *energy;
    float *spectrum;
    float flux;


    __host__ void init(Settings &s) {
        Settings::init(s.tube, s.voltage, s.power, s.det_resolution, s.det_size, s.exposure);

        // file paths
        std::string tube_str = tube2str(tube);
        std::string root_path = std::string("./data/tubes/X-Ray W11D ") + tube_str + "/";
        std::string flux_path = root_path + tube_str + " Fg 100 cm 1 kW.txt";
        std::string spectrum_path = root_path + "W11D " + std::to_string(int(voltage)) + " kV " + tube_str + ".txt";
        
        // load spectrum and flux data
        std::vector<float> energy_vec, spectrum_vec;
        read_data(spectrum_path, energy_vec, spectrum_vec);
        std::vector<float> energy_vec_fl, flux_vec;
        read_data(flux_path, energy_vec_fl, flux_vec);

        // move spectrum and energy from RAM to GPU memory
        cudaMallocManaged(&energy, energy_vec.size()*sizeof(float));
        cudaMallocManaged(&spectrum, energy_vec.size()*sizeof(float));
        copy(energy_vec.begin(), energy_vec.end(), energy);
        copy(spectrum_vec.begin(), spectrum_vec.end(), spectrum);

        // find appropriate flux and scale it
        flux = -1.0;
        for(int i = 0; i < energy_vec_fl.size(); i++) {
            if(abs(voltage - energy_vec_fl[i]) < FLT_EPS) {
                flux = flux_vec[i] * power;
                break;
            }
        }
        if(abs(flux - (-1.0)) < FLT_EPS) {
            std::string msg = "There is no appropriate value of flux for the given voltage.\n";
            std::cerr << msg;
            throw std::invalid_argument(msg);
        }
    }

    __host__ ~SettingsDevice() {
        cudaDeviceSynchronize();
        printf("settingsDevice destructor\n");
        cudaFree(energy);
        cudaFree(spectrum);
    }
};

#endif  // SETTINGS_DEVICE_CUH