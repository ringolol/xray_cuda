#ifndef MATRIX_CUH
#define MATRIX_CUH


struct Matrix {
    /**
     * The matrix width
     */
    int width;
    /**
     * The matrix height
     */
    int height;
    /**
     * The pixels' density (pixels per cm)
     */
    float density;
    /**
     * The matrix itself represented as 3d points
     */
    float3 **cells;
    /**
     * The image formed on the matrix by the x-rays
     */
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
        
        // calculate pixel location from its index and matrix size
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