#ifndef UTILS_HOST_H
#define UTILS_HOST_H


#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "types.h"

#define FLT_EPS 0.00001 // used for floats comparison
#define SQR(X) ((X) * (X)) // square of X


/**
 * Read X-Y data from the file
 * @param path file path
 * @param x_data returned X-data
 * @param y_data returned Y-data
 * 
 * @throw std::ios_base::failure
**/
void read_data(std::string path, std::vector<float> &x_data, std::vector<float> &y_data) {
    std::ifstream infile;
    infile.exceptions(std::ifstream::failbit);
    
    try {
        infile.open(path, std::ifstream::in);
    } catch(std::ios_base::failure& fail) {
        std::cerr << "Opening file '" << path 
            << "' failed, it either doesn't exist or is not accessible.\n";
        throw;
    }

    try {
        float xi, yi;

        while (infile >> xi >> yi) {
            x_data.push_back(xi);
            y_data.push_back(yi);
        }
    } catch (...) {}

    infile.close();
}

#endif  // UTILS_HOST_H
