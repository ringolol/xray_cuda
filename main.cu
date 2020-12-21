/**
* The CUDA x-ray demo program which calculates an x-ray image.
*
* @author  Valeriy Lyubich
* @version 0.1
* @since   2020-12-18
*/

#include <math.h>
#include <iostream>

#include <cuda_profiler_api.h>

#include "utils.cuh"
#include "matrix.cuh"
#include "beam.cuh"
#include "block.cuh"
#include "f3_overload.cuh"
#include "xray_calc.cuh"


/*
    CUDA commands.

    build:
        nvcc -o ./build/app.exe ./main.cu -arch=sm_61
    run:
        nvprof ./build/app.exe
    memory check (build with flags -G and -g):
        cuda-memcheck .\build\app.exe |more
*/


int main() {
    // load data
    float *E=nullptr, *DQE=nullptr, *nuAir=nullptr, *nuIron=nullptr;
    read_data("./data.txt", E, DQE, nuAir, nuIron);

    // x-ray source
    float3 source = make_float3(0.0, 0.0, 0.0);
    // blocks representing 3d objects
    Block* blocks;
    // sensor matrix
    Matrix* matrix;

    // allocate memorry in managed memory
    int blocks_num = 6;
    cudaMallocManaged(&blocks, blocks_num*sizeof(Block));
    cudaMallocManaged(&matrix, 1*sizeof(Matrix));
    float3 *block1_points, *block2_points, *block3_points, *block4_points, *block5_points, *block6_points;
    cudaMallocManaged(&block1_points, 4*sizeof(float3));
    cudaMallocManaged(&block2_points, 4*sizeof(float3));
    cudaMallocManaged(&block3_points, 4*sizeof(float3));
    cudaMallocManaged(&block4_points, 4*sizeof(float3));
    cudaMallocManaged(&block5_points, 4*sizeof(float3));
    cudaMallocManaged(&block6_points, 4*sizeof(float3));

    // init blocks
    float p_hsize = 4.5; //4.5 (edge case)
    float p_z = -35;
    float p1_thicc = 2, p2_thicc = 2;
    float hole_size = 1;
    float hh = hole_size/2;
    // init first block
    block1_points[0] = make_float3(-p_hsize, -p_hsize, p_z);
    block1_points[1] = make_float3( p_hsize, -p_hsize, p_z);
    block1_points[2] = make_float3(-p_hsize,  p_hsize, p_z);
    block1_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc);
    blocks[0].init(block1_points, iron);


    block2_points[0] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc);
    block2_points[1] = make_float3(-hh,      -p_hsize, p_z-p1_thicc);
    block2_points[2] = make_float3(-p_hsize, p_hsize,  p_z-p1_thicc);
    block2_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    blocks[1].init(block2_points, iron);

    block3_points[0] = make_float3(hh,      -p_hsize, p_z-p1_thicc);
    block3_points[1] = make_float3(p_hsize, -p_hsize, p_z-p1_thicc);
    block3_points[2] = make_float3(hh,      p_hsize,  p_z-p1_thicc);
    block3_points[3] = make_float3(hh,      -p_hsize, p_z-p1_thicc-hole_size);
    blocks[2].init(block3_points, iron);

    block4_points[0] = make_float3(-hh, -p_hsize, p_z-p1_thicc);
    block4_points[1] = make_float3(hh,  -p_hsize, p_z-p1_thicc);
    block4_points[2] = make_float3(-hh, -hh,      p_z-p1_thicc);
    block4_points[3] = make_float3(-hh, -p_hsize, p_z-p1_thicc-hole_size);
    blocks[3].init(block4_points, iron);

    block5_points[0] = make_float3(-hh, hh,      p_z-p1_thicc);
    block5_points[1] = make_float3(hh,  hh,      p_z-p1_thicc);
    block5_points[2] = make_float3(-hh, p_hsize, p_z-p1_thicc);
    block5_points[3] = make_float3(-hh, hh,      p_z-p1_thicc-hole_size);
    blocks[4].init(block5_points, iron);


    block6_points[0] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    block6_points[1] = make_float3( p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    block6_points[2] = make_float3(-p_hsize,  p_hsize, p_z-p1_thicc-hole_size);
    block6_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size-p2_thicc);
    blocks[5].init(block6_points, iron);

    // init matrix
    float matrix_width = 40;
    int matrix_width_px = 1024;
    int matrix_height_px = 1024;
    matrix->init(matrix_width_px, matrix_height_px, matrix_width/matrix_width_px, -90.0);
    printf("matrix size: %dx%d\n", matrix->width, matrix->height);

    // start x-ray image calculation
    int threads_size = 32;
    dim3 threadsPerBlock(threads_size, threads_size);
    int blocks_width = ceil((float)matrix_width_px/threads_size);
    int blocks_height = ceil((float)matrix_height_px/threads_size);
    dim3 blocksShape(blocks_width, blocks_height);

    // run kernel to calculate x-ray image
    calc_xray_image<<<blocksShape, threadsPerBlock>>>(source, blocks, blocks_num, matrix);

    // wait for all threads and blocks
    cudaDeviceSynchronize();

    std::ofstream outdata("output.txt");
    for(int i = 0; i < matrix->width; i++) {
        for(int j = 0; j < matrix->height; j++) {
            outdata << matrix->image[i][j];
            if(j != matrix->height-1)
                outdata << '\t';
        }
        if(i != matrix->width-1)
            outdata << '\n';
    }
    outdata.close();

    // print errors
    gpuErrchk( cudaPeekAtLastError() );

    // free allocated memory
    cudaFree(blocks);
    cudaFree(matrix);
    cudaFree(block1_points);
    cudaFree(block2_points);

    cudaProfilerStop();
    return 0;
}