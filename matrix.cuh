#ifndef MATRIX_CUH
#define MATRIX_CUH

/**
* X-ray detector's matrix
*/
struct Matrix {
    int width, height;
    float density;
    float3 **cells;

    __host__ void init(int w, int h, float d) {
        width = w;
        height = h;
        density = d;
        gpuErrchk( cudaMallocManaged(&cells, width*sizeof(float3*)) );
        for(int i = 0; i < width; i++) {
            gpuErrchk( cudaMallocManaged(&cells[i], height*sizeof(float3*)) );
        }
        
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                cells[i][j] = make_float3(
                    density * (i - width/2 + 0.5), 
                    density * (j - height/2 + 0.5), 
                    -90
                );
            }
        }
    }
};

#endif