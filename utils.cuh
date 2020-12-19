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

__device__ __host__ void print_float3(float3 val, const char* str = "val") {
    printf("%s = (%f, %f, %f)\n", str, val.x, val.y, val.z);
}

__device__ __host__ float f3_dist(float3 a, float3 b) {
    return sqrtf(SQR(a.x-b.x) + SQR(a.y-b.y) + SQR(a.z-b.z));
}

__device__ __host__ float3 nanf3() {
    return float3(make_float3(nanf(""), nanf(""), nanf("")));
}

/**
* Read data from a file
*/
void read_data(std::string path, float *e, float *dqe, float *sigm) {
    std::ifstream infile(path);
    float e_, dqe_, sigm_;
    int N;
    infile >> N;
    e = new float[N];
    dqe = new float[N];
    sigm = new float[N];
    int i = 0;
    while (infile >> e_ >> dqe_ >> sigm_) {
        printf("%f %f %f\n", e_, dqe_, sigm_);
        e[i] = e_;
        dqe[i] = dqe_;
        sigm[i] = sigm_;
        i++;
    }
}

#endif