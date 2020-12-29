#ifndef UTILS_DEVICE_CUH
#define UTILS_DEVICE_CUH


/** 
 * CUDA error check
 * @param ans a cuda function return
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif  // UTILS_DEVICE_CUH