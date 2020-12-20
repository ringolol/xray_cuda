/**
* The CUDA x-ray demo program which calculates x-ray image.
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


/**
* Intersect a beam and a plane
*/
__device__ float3 intersect_plane(Plane plane, Beam beam) {
    float d = (plane.points[0] - beam.l0) * plane.normal / (beam.l * plane.normal);
    float3 p = beam.l0 + beam.l*d;
    
    float aa = (plane.points[1] - plane.points[0])*(plane.points[1] - plane.points[0]);
    float ap = (plane.points[1] - plane.points[0])*(p - plane.points[0]);
    if(ap <= aa && ap >= 0) {
        float bb = (plane.points[2] - plane.points[0])*(plane.points[2] - plane.points[0]);
        float bp = (plane.points[2] - plane.points[0])*(p - plane.points[0]);
        if(bp <= bb && bp >= 0) {
            return p;
        }
    }
    
    return nanf3();
}

/**
* Find all intersections between a beam and blocks
*/
__device__ void intersect(Block* blocks, int blocks_num, Beam beam) {
    float *lens = new float[blocks_num];

    for(int i = 0; i < blocks_num; i++) {
        int ic = 0;
        float3 inters[2] = {nanf3(), nanf3()};
        for(int j = 0; j < 6; j++) {
            float3 inter =  intersect_plane(blocks[i].planes[j], beam);
            
            if(!isnan(inter.x)) {
                inters[ic] = inter;
                ic++;
                if(ic == 2) break;
            }
        }
        float d = 0.0;
        if(!isnan(inters[0].x) && !isnan(inters[1].x) ) {
            d = f3_dist(inters[0], inters[1]);
        } else {
            d = 0.0;
        }
        lens[i] = d;

        // printf("LEN{beam[%d][%d]/block[%i]} = %f\n", beam.inx.x, beam.inx.y, i, lens[i]);
    }
    delete[] lens;
}

/**
* Calcualte resulting x-ray image on the detector's matrix
*/ 
__global__ void calc_xray_image(float3 source, Block* blocks, int blocks_num, Matrix* matrix) {
    int idX = threadIdx.x+blockDim.x*blockIdx.x;
    int idY = threadIdx.y+blockDim.y*blockIdx.y;
    if(idX < matrix->width && idY < matrix->height) {
        float3 cell = matrix->cells[idX][idY];
        Beam beam(source, cell, make_int2(idX, idY));
        intersect(blocks, blocks_num, beam);
    }
}


int main() {
    // load data
    // float *E=nullptr, *DQE=nullptr, *nuAir=nullptr, *nuIron=nullptr;
    // read_data("./data.txt", E, DQE, nuAir, nuIron);

    // x-ray source
    float3 source = make_float3(0.0, 0.0, 0.0);
    // blocks representing 3d objects
    Block* blocks;
    // sensor matrix
    Matrix* matrix;

    // allocate memorry in managed memory
    int blocks_num = 2;
    cudaMallocManaged(&blocks, blocks_num*sizeof(Block));
    cudaMallocManaged(&matrix, 1*sizeof(Matrix));
    float3 *block1_points, *block2_points;
    cudaMallocManaged(&block1_points, 4*sizeof(float3));
    cudaMallocManaged(&block2_points, 4*sizeof(float3));

    // init blocks
    float p_hsize = 10; //4.5 (edge case)
    float p_z = -40;
    float p_thicc = 5;
    // init first block
    block1_points[0] = make_float3(-p_hsize, -p_hsize, p_z);
    block1_points[1] = make_float3( p_hsize, -p_hsize, p_z);
    block1_points[2] = make_float3(-p_hsize,  p_hsize, p_z);
    block1_points[3] = make_float3(-p_hsize, -p_hsize, p_z - p_thicc);
    blocks[0].init(block1_points, iron);
    // init second block
    block2_points[0] = make_float3(-p_hsize, -p_hsize, p_z - p_thicc);
    block2_points[1] = make_float3( p_hsize, -p_hsize, p_z - p_thicc);
    block2_points[2] = make_float3(-p_hsize,  0,       p_z - p_thicc);
    block2_points[3] = make_float3(-p_hsize, -p_hsize, p_z - 2*p_thicc);
    blocks[1].init(block2_points, iron);

    // init matrix
    float matrix_width = 40;
    int matrix_width_px = 1024;
    int matrix_height_px = 1024;
    matrix->init(matrix_width_px, matrix_height_px, matrix_width_px/matrix_width);
    printf("matrix size: %dx%d\n", matrix->width, matrix->height);

    // start x-ray image calculation
    int threads_size = 32;
    dim3 threadsPerBlock(threads_size, threads_size);
    int blocks_width = ceil((float)matrix_width_px/threads_size);
    int blocks_height = ceil((float)matrix_height_px/threads_size);
    dim3 blocksShape(blocks_width, blocks_height);

    float *lens;
    cudaMallocManaged(&lens, blocks_num*sizeof(float));

    calc_xray_image<<<blocksShape, threadsPerBlock>>>(source, blocks, blocks_num, matrix);
    // wait for all threads and blocks
    cudaDeviceSynchronize();
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