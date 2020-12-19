#ifndef F3_OVERLOAD_CUH
#define F3_OVERLOAD_CUH

#include "utils.cuh"


/**
* Sum of two vectors
*/
__device__ __host__ float3 operator+(float3 a, const float3& b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

/**
* Sub of two vectors
*/
__device__ __host__ float3 operator-(float3 a, const float3& b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

/**
* Multiply a vector by a constant
*/
__device__ __host__ float operator*(float3 a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
* Dot product of two vectors
*/
__device__ __host__ float3 operator*(float3 a, const float& b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

/**
* Cross product of two vectors
* source: https://en.wikipedia.org/wiki/Cross_product#Alternative_ways_to_compute_the_cross_product
*/
__device__ __host__ float3 operator%(float3 a, const float3& b) {
    // 
    return make_float3(
        -a.z * b.y + a.y * b.z,
        a.z * b.x - a.x * b.z,
        -a.y * b.x + a.x * b.y
    );
}

/**
* Equality of two vectors
*/
__device__ __host__ bool operator==(float3 a, const float3& b) {
    return fabsf(a.x-b.x) < FLT_EPS && fabsf(a.y-b.y) < FLT_EPS && fabsf(a.z-b.z) < FLT_EPS;
}

/**
* Inequality of two vectors
*/
__device__ __host__ bool operator!=(float3 a, const float3& b) {
    return !(a == b);
}

#endif

