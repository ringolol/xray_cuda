#ifndef UTILS_CUH
#define UTILS_CUH

#include <string>
#include <fstream>


#define FLT_EPS 0.00001 // used for floats comparison
#define SQR(X) ((X) * (X)) // square of X
#define LEN(X) (sizeof(X)/sizeof(*X)) // ain't workin' in cuda;(


enum Material {
    air = 0,
    iron = 1,
};


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
void read_data(std::string path, float *e, float *dqe, float *nuAir, float *nuIron) {
    std::ifstream infile(path);
    float e_, dqe_, nuAir_, nuIron_;
    int N;
    infile >> N;
    e = new float[N];
    dqe = new float[N];
    nuAir = new float[N];
    nuIron = new float[N];
    int i = 0;
    while (infile >> e_ >> dqe_ >> nuAir_ >> nuIron_) {
        // printf("%f %f %f %f\n", e_, dqe_, nuAir_, nuIron_);
        e[i] = e_;
        dqe[i] = dqe_;
        nuAir[i] = nuAir_;
        nuIron[i] = nuIron_;
        i++;
    }
}

#endif