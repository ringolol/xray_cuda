/**
* The CUDA x-ray demo program which calculates x-ray image.
*
* @author  Valeriy Lyubich
* @version 0.1
* @since   2020-12-18
*/

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
            float3 inter = intersect_plane(blocks[i].planes[j], beam);
            
            if(!isnan(inter.x)) {
                // printf("intersect in beam[%d][%d]/block[%d]/plane[%d]: (%f, %f, %f)\n", beam.inx.x, beam.inx.y, i, j, inter.x, inter.y, inter.z);
                inters[ic] = inter;
                ic++;
                if(ic == 2) break;
            }
        }
        float d = 0.0;
        if(!isnan(inters[0].x) && !isnan(inters[1].x) ) {
            d = f3_dist(inters[0], inters[1]);
            // printf("dist[(%f, %f, %f), (%f, %f, %f)] = %f\n", inters[0].x, inters[0].y, inters[0].z, inters[1].x, inters[1].y, inters[1].z, d);
        } else {
            d = 0.0;
        }
        lens[i] = d;
        // printf("len[%d] = %f\n", i, lens[i]);
    }
}

/**
* Calcualte resulting x-ray image on the detector's matrix
*/ 
__global__ void calc_xray_image(float3 source, Block* blocks, int blocks_num, Matrix* matrix) {
    float3 cell = matrix->cells[threadIdx.x][threadIdx.y];
    Beam beam(source, cell, make_int2(threadIdx.x, threadIdx.y));
    // print_float3(source, "source");
    // print_float3(cell, "cell");
    intersect(blocks, blocks_num, beam);
}


int main() {
    float *E=nullptr, *DQE=nullptr, *Sigm=nullptr;
    read_data("./data.txt", E, DQE, Sigm);

    float3 source = make_float3(0.0, 0.0, 0.0);
    Block* blocks;
    Matrix* matrix;
    int blocks_num = 2;
    gpuErrchk( cudaMallocManaged(&blocks, blocks_num*sizeof(Block)) );
    gpuErrchk( cudaMallocManaged(&matrix, 1*sizeof(Matrix)) );

    float3 *plane1_points, *plane2_points;
    gpuErrchk( cudaMallocManaged(&plane1_points, 4*sizeof(float3)) );
    gpuErrchk( cudaMallocManaged(&plane2_points, 4*sizeof(float3)) );
    float p_hsize = 10; //4.5 (edge case)
    float p_z = -40;
    float p_thicc = 5;
    plane1_points[0] = make_float3(-p_hsize, -p_hsize, p_z);
    plane1_points[1] = make_float3( p_hsize, -p_hsize, p_z);
    plane1_points[2] = make_float3(-p_hsize,  p_hsize, p_z);
    plane1_points[3] = make_float3(-p_hsize, -p_hsize, p_z - p_thicc);
    blocks[0].init(plane1_points, iron);

    plane2_points[0] = make_float3(-p_hsize, -p_hsize, p_z - p_thicc);
    plane2_points[1] = make_float3( p_hsize, -p_hsize, p_z - p_thicc);
    plane2_points[2] = make_float3(-p_hsize,  0,       p_z - p_thicc);
    plane2_points[3] = make_float3(-p_hsize, -p_hsize, p_z - 2*p_thicc);
    blocks[1].init(plane2_points, iron);

    float overall_size = 40;
    int pixel_num = 32;
    matrix->init(pixel_num, pixel_num, overall_size/pixel_num);

    printf("matrix size: %dx%d\n", matrix->width, matrix->height);

    dim3 threadsPerBlock(matrix->width, matrix->height);
    calc_xray_image<<<1, threadsPerBlock>>>(source, blocks, blocks_num, matrix);
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    cudaFree(blocks);
    cudaFree(matrix);
    cudaFree(plane1_points);
    cudaFree(plane2_points);

    return 0;
}