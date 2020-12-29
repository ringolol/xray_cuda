#ifndef BLOCK_CUH
#define BLOCK_CUH


#include "../utils/f3_overload.cuh"


/**
* A plane (rectangle) in 3d space
*/
struct Plane {
    float3 normal;
    float3 points[3];

    /**
     * Initialize the plane using three dots
     * @param p0 the central point
     * @param p1 first corner point
     * @param p2 second corner point
     */
    __host__ void init(float3 p0, float3 p1, float3 p2) {
        // p0 must be a central point!
        // p1 and p2 must be corner points
        // 
        // p0 *----------* p1
        //    |          |
        // p2 *----------o

        // cudaMallocManaged(&points, 3*sizeof(float3));
        points[0] = p0;
        points[1] = p1;
        points[2] = p2;

        // normal to the plane
        // https://en.wikipedia.org/wiki/Plane_(geometry)#Method_3
        normal = (points[1] - points[0]) % (points[2] - points[0]);
    }

    #ifdef __CUDA_ARCH__
        __device__ ~Plane() {
            delete[] points;
        }
    #else
        __host__ ~Plane() {
            cudaFree(points);
        }
    #endif
};

/**
* A Cuboid
*/
struct Block {
    Plane *planes;
    Material material;
    float *mean_path;
    
    /**
     * Initialize the block (cuboid) using four points and material data
     * @param points_ four points of the cuboid (in some particular order!)
     * @param material material of the block (not in use)
     * @param mean_path_ material physics
     */
    __host__ void init(float3 points_[4], Material material_, float *mean_path_) {
        // a cuboid can be defined by 4 points and the material it
        // consists of
        material = material_;
        mean_path = mean_path_;

        float3 points[7];
        // gpuErrchk( cudaMallocManaged(&points, 7*sizeof(float3)) );
        cudaMallocManaged(&planes, 6*sizeof(Plane));
        // main points
        points[0] = points_[0];
        points[1] = points_[1];
        points[2] = points_[2];
        points[3] = points_[3];
        // additional points can be constructed from main points
        points[4] = points[1] + (points[3] - points[0]);
        points[5] = points[2] + (points[3] - points[0]);
        points[6] = points[1] + (points[2] - points[0]);

        // cuboid consists of 6 planes
        planes[0].init(points[0], points[1], points[2]);
        planes[1].init(points[0], points[2], points[3]);
        planes[2].init(points[0], points[1], points[3]);
        planes[3].init(points[3], points[4], points[5]);
        planes[4].init(points[2], points[5], points[6]);
        planes[5].init(points[1], points[4], points[6]);
    }

    #ifdef __CUDA_ARCH__
        __device__ ~Block() {
            delete[] planes;
        }
    #else
        __host__ ~Block() {
            cudaFree(planes);
        }
    #endif
    
};

#endif  // BLOCK_CUH

