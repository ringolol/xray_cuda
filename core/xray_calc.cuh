#ifndef XRAY_CALC_CUH
#define XRAY_CALC_CUH

#define _USE_MATH_DEFINES
#include <math.h>

#include <curand_kernel.h>

#include "../utils/utils_host.h"
#include "../utils/f3_overload.cuh"
#include "matrix.cuh"
#include "beam.cuh"
#include "block.cuh"
#include "settings_device.cuh"


/**
 * Intersect the beam and the plane
 * @param plane the plane to intersect
 * @param beam the beam to intersect
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
 * Find all intersections between the beam and the given blocks
 * @param blocks blocks representing parts in 3d space
 * @param blocks_num number of blocks
 * @param beam the beam from source to the specific matrix pixel
 * @param energy_inx the current spectrum energy index
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

/**
 * Calcualte clear (without noise) x-ray image of the pixel ("idX", "idY") for quant energy "energy_inx"
 * @param source source location
 * @param blocks blocks representing parts
 * @param blocks_num number of blocks
 * @param matrix sensor's matrix
 * @param settings x-ray imager settings
 * @param idX x-index of the matrix cell
 * @param idY y-index of the matrix cell
 * @param I0i number of quants which would hit detector (if there would be no obstacles)
 * @param energy_inx the current spectrum energy index
 */
__device__ void calc_xray_group(float3 source, Block* blocks, int blocks_num, Matrix* matrix, SettingsDevice *settings, int idX, int idY, float I0i, int energy_inx) {
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

/**
 * Calculate x-ray image of the pixel ("idX", "idY")
 * @param source source location
 * @param blocks blocks representing parts
 * @param blocks_num number of blocks
 * @param matrix sensor's matrix
 * @param settings x-ray imager settings
 * @param idX x-index of the matrix cell
 * @param idY y-index of the matrix cell
 */
__device__ void calc_xray_image(float3 source, Block* blocks, int blocks_num, Matrix* matrix, SettingsDevice *settings, curandState *global_state, int idX, int idY) {
    float T = settings->exposure;
    for(int energy_inx = 0; energy_inx < settings->voltage; energy_inx++) {
        // temp detector value!
        float epsi = 1;

        float I0i = settings->flux * (settings->spectrum[energy_inx] / 100) * T * epsi;
        calc_xray_group(source, blocks, blocks_num, matrix, settings, idX, idY, I0i, energy_inx);
    }

    // calculate noise
    // temp detector value!
    float B = 0; //powf(10, -15);
    // temp electronic noise
    float Nn = 5;
    float Npi = matrix->image[idX][idY];
    
    // noise normal distribution parameters
    float mean = 0.0;
    float std_dev = sqrtf(Npi + B * SQR(Npi) + SQR(Nn));
    
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = global_state[id];
    // use build-in normal distribution (with mean 0 and std 1) to estimate noise
    matrix->image[idX][idY] += (curand_normal(&local_state) + mean) * std_dev;
    global_state[id] = local_state;
}

/**
 * Setting up the global cuRAND state for noise generation
 * source: https://stackoverflow.com/a/14291364/11162245
 * @param state array of cuRAND global states to initialize
 */
__global__ void setup_curand(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(7+id, id, 0, &state[id]);
}

/**
 * Calcualte resulting x-ray image on the detector's matrix (cuda kernel entry point)
 * @param source source location
 * @param blocks blocks representing parts
 * @param blocks_num number of blocks
 * @param matrix sensor's matrix
 * @param settings x-ray imager settings
 * @param global_state cuRAND global state (is used for random noise generation)
 */ 
__global__ void xray_image_kernel(float3 source, Block* blocks, int blocks_num, Matrix* matrix, SettingsDevice *settings, curandState *global_state) {
    int idX = threadIdx.x+blockDim.x*blockIdx.x;
    int idY = threadIdx.y+blockDim.y*blockIdx.y;

    if(idX < matrix->width && idY < matrix->height) {
        calc_xray_image(source, blocks, blocks_num, matrix, settings, global_state, idX, idY);
    }
}

#endif  // XRAY_CALC_CUH