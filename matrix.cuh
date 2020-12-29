#ifndef MATRIX_CUH
#define MATRIX_CUH

/**
* X-ray detector's matrix
*/
struct Matrix {
    int width, height;
    float density;
    float3 **cells;
    float **image;

    /**
     * Initialize detector's matrix (parallel to XOY plane)
     * @param w matrix phisical width
     * @param h matrix phisical height
     * @param d pixels' density (number of pixels per cm)
     * @param z Z coordinate of the detector
     */
    __host__ void init(int w, int h, float d, float z) {
        width = w;
        height = h;
        density = d;
        cudaMallocManaged(&cells, width*sizeof(float3*));
        cudaMallocManaged(&image, width*sizeof(float*));
        for(int i = 0; i < width; i++) {
            cudaMallocManaged(&cells[i], height*sizeof(float3));
            cudaMallocManaged(&image[i], height*sizeof(float));
        }
        
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                cells[i][j] = make_float3(
                    density * (i - width/2 + 0.5), 
                    density * (j - height/2 + 0.5), 
                    z
                );
            }
        }
    }
};

#endif  // MATRIX_CUH