#ifndef XRAY_CALC_CUH
#define XRAY_CALC_CUH

#define _USE_MATH_DEFINES
#include <math.h>

#include "utils.cuh"
#include "matrix.cuh"
#include "beam.cuh"
#include "block.cuh"
#include "f3_overload.cuh"
#include "settings.cuh"


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
__device__ float intersect(Block* blocks, int blocks_num, Beam beam, int energy_inx) {
    float Ki = 0.0f;

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
        float l = 0.0;
        if(!isnan(inters[0].x) && !isnan(inters[1].x) ) {
            l = f3_dist(inters[0], inters[1]);
        } else {
            l = 0.0;
        }

        Ki += l / blocks[i].mean_path[energy_inx];
    }

    return Ki;
}

__device__ void calc_xray_group(float3 source, Block* blocks, int blocks_num, Matrix* matrix, Settings *settings, int idX, int idY, float I0i, int energy_inx) {
    float3 cell = matrix->cells[idX][idY];
    Beam beam(source, cell, make_int2(idX, idY));
    float Ki = intersect(blocks, blocks_num, beam, energy_inx);
    float Ii_p = I0i*expf(-Ki);
    
    // angle between the plane and the beam
    // source: https://www.superprof.co.uk/resources/academic/maths/analytical-geometry/distance/angle-between-line-and-plane.html
    float3 beam_vec = cell - source;
    float3 cell_normal = make_float3(0,0,1);
    float sina = fabsf(cell_normal * beam_vec) / (norm2(cell_normal) * norm2(beam_vec));
    float dSp_s = SQR(matrix->density)*sina;

    float Ii = Ii_p * dSp_s * SQR(100.0/beam.len);

    matrix->image[idX][idY] += Ii;
}

__device__ void calc_xray_image(float3 source, Block* blocks, int blocks_num, Matrix* matrix, Settings *settings, int idX, int idY) {
    // temp values
    for(int energy_inx = 0; energy_inx < settings->voltage; energy_inx++) {
        // temp detector values
        float epsi = 1.0;
        float T = 1;

        float I0i = settings->flux * (settings->spectrum[energy_inx] / 100) * T * epsi;

        calc_xray_group(source, blocks, blocks_num, matrix, settings, idX, idY, I0i, energy_inx);
    }

}

/**
* Calcualte resulting x-ray image on the detector's matrix
*/ 
__global__ void xray_image_kernel(float3 source, Block* blocks, int blocks_num, Matrix* matrix, Settings *settings) {
    int idX = threadIdx.x+blockDim.x*blockIdx.x;
    int idY = threadIdx.y+blockDim.y*blockIdx.y;

    if(idX < matrix->width && idY < matrix->height) {
        calc_xray_image(source, blocks, blocks_num, matrix, settings, idX, idY);
    }
}

#endif