#ifndef UTILS_CUH
#define UTILS_CUH

#include <string>
#include <fstream>
#include <vector>

#include "types.h"


#define FLT_EPS 0.00001 // used for floats comparison
#define SQR(X) ((X) * (X)) // square of X
#define LEN(X) (sizeof(X)/sizeof(*X)) // ain't workin' in cuda;(


// CUDA error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
* Read data from a file
*/
void read_data(std::string path, std::vector<float> &x_data, std::vector<float> &y_data) {
    std::ifstream infile;
    infile.exceptions(std::ifstream::failbit);
    
    try {
        infile.open(path, std::ifstream::in);
        // infile.close();
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
    
}

#endif