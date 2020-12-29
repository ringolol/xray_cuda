#ifndef BEAM_CUH
#define BEAM_CUH

#include "f3_overload.cuh"


/**
* X-ray beam, it starts from source and ends on a matrix' cell
*/
struct Beam {
    float3 p0, p1;
    float3 l0, l;
    int2 inx;
    float len;

    __device__ Beam(float3 p0_, float3 p1_, int2 inx_) {
        p0 = p0_;
        p1 = p1_;
        inx = inx_;
        len = f3_dist(p0, p1);

        // pars for intersection with planes
        l0 = p0;
        l = p1 - p0;
    }
};

#endif  // BEAM_CUH

