#include <math.h>
#include <iostream>
#include <vector>

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>

#include "utils.cuh"
#include "matrix.cuh"
#include "beam.cuh"
#include "block.cuh"
#include "f3_overload.cuh"
#include "xray_calc.cuh"
#include "settings.cuh"

#include "xray_imager.h"


float* load_iron_data(Settings* settings) {
    // load materials' data
    std::vector<float> energy_vec, mean_path_vec;
    read_data("./data/materials/G_Fe.txt", energy_vec, mean_path_vec);
    float *Fe_x;
    cudaMallocManaged(&Fe_x, mean_path_vec.size()*sizeof(float));
    copy(mean_path_vec.begin(), mean_path_vec.end(), Fe_x);
    // interpolate mean path for Fe
    int k = 0;
    for(int i = 0; i < settings->voltage; i++) {
        while(settings->energy[i] >= energy_vec[k]) {
            k++;
        }
        float mean_path_k = (mean_path_vec[k] - mean_path_vec[k-1]) / (energy_vec[k] - energy_vec[k-1]);
        Fe_x[i] = mean_path_vec[k-1] + (settings->energy[i] - energy_vec[k-1]) * mean_path_k;
    }

    return Fe_x;
}

void plate_with_holl(std::vector<Block> &blocks, Settings* settings, float hole_size, float p_thicc) {
    float *Fe_x = load_iron_data(settings);

    float p1_thicc = (p_thicc-hole_size)/2;
    float p2_thicc = p1_thicc;
    
    // allocate memorry in managed memory
    blocks.resize(6);
    float3 *block1_points, *block2_points, *block3_points, *block4_points, *block5_points, *block6_points;
    cudaMallocManaged(&block1_points, 4*sizeof(float3));
    cudaMallocManaged(&block2_points, 4*sizeof(float3));
    cudaMallocManaged(&block3_points, 4*sizeof(float3));
    cudaMallocManaged(&block4_points, 4*sizeof(float3));
    cudaMallocManaged(&block5_points, 4*sizeof(float3));
    cudaMallocManaged(&block6_points, 4*sizeof(float3));

    // init blocks
    float p_hsize = 70; //4.5 (edge case)
    float p_z = -35;
    float hh = hole_size/2;

    // first layer
    block1_points[0] = make_float3(-p_hsize, -p_hsize, p_z);
    block1_points[1] = make_float3( p_hsize, -p_hsize, p_z);
    block1_points[2] = make_float3(-p_hsize,  p_hsize, p_z);
    block1_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc);
    (blocks[0]).init(block1_points, iron, Fe_x);

    // second layer with the hole
    block2_points[0] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc);
    block2_points[1] = make_float3(-hh,      -p_hsize, p_z-p1_thicc);
    block2_points[2] = make_float3(-p_hsize, p_hsize,  p_z-p1_thicc);
    block2_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    blocks[1].init(block2_points, iron, Fe_x);

    block3_points[0] = make_float3(hh,      -p_hsize, p_z-p1_thicc);
    block3_points[1] = make_float3(p_hsize, -p_hsize, p_z-p1_thicc);
    block3_points[2] = make_float3(hh,      p_hsize,  p_z-p1_thicc);
    block3_points[3] = make_float3(hh,      -p_hsize, p_z-p1_thicc-hole_size);
    blocks[2].init(block3_points, iron, Fe_x);

    block4_points[0] = make_float3(-hh, -p_hsize, p_z-p1_thicc);
    block4_points[1] = make_float3(hh,  -p_hsize, p_z-p1_thicc);
    block4_points[2] = make_float3(-hh, -hh,      p_z-p1_thicc);
    block4_points[3] = make_float3(-hh, -p_hsize, p_z-p1_thicc-hole_size);
    blocks[3].init(block4_points, iron, Fe_x);

    block5_points[0] = make_float3(-hh, hh,      p_z-p1_thicc);
    block5_points[1] = make_float3(hh,  hh,      p_z-p1_thicc);
    block5_points[2] = make_float3(-hh, p_hsize, p_z-p1_thicc);
    block5_points[3] = make_float3(-hh, hh,      p_z-p1_thicc-hole_size);
    blocks[4].init(block5_points, iron, Fe_x);

    // third layer
    block6_points[0] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    block6_points[1] = make_float3( p_hsize, -p_hsize, p_z-p1_thicc-hole_size);
    block6_points[2] = make_float3(-p_hsize,  p_hsize, p_z-p1_thicc-hole_size);
    block6_points[3] = make_float3(-p_hsize, -p_hsize, p_z-p1_thicc-hole_size-p2_thicc);
    blocks[5].init(block6_points, iron, Fe_x);
}

void plate_with_notch(std::vector<Block> &blocks, Settings* settings, float notch_size, float notch_depth, float plate_thicc) {
    float *Fe_x = load_iron_data(settings);

    blocks.resize(3);
    float3 *block1_points, *block2_points, *block3_points;
    cudaMallocManaged(&block1_points, 4*sizeof(float3));
    cudaMallocManaged(&block2_points, 4*sizeof(float3));
    cudaMallocManaged(&block3_points, 4*sizeof(float3));

    // init blocks
    float p_hsize = 70; //4.5 (edge case)
    float p_z = -35;
    float hh = notch_size/2;
    
    block1_points[0] = make_float3(-p_hsize, -p_hsize, p_z-notch_depth);
    block1_points[1] = make_float3( p_hsize, -p_hsize, p_z-notch_depth);
    block1_points[2] = make_float3(-p_hsize,  p_hsize, p_z-notch_depth);
    block1_points[3] = make_float3(-p_hsize, -p_hsize, p_z-(plate_thicc-notch_depth));
    (blocks[0]).init(block1_points, iron, Fe_x);

    // second layer with the hole
    block2_points[0] = make_float3(-p_hsize, -p_hsize, p_z);
    block2_points[1] = make_float3(-hh,      -p_hsize, p_z);
    block2_points[2] = make_float3(-p_hsize, p_hsize,  p_z);
    block2_points[3] = make_float3(-p_hsize, -p_hsize, p_z-notch_depth);
    blocks[1].init(block2_points, iron, Fe_x);

    block3_points[0] = make_float3(hh,      -p_hsize, p_z);
    block3_points[1] = make_float3(p_hsize, -p_hsize, p_z);
    block3_points[2] = make_float3(hh,      p_hsize,  p_z);
    block3_points[3] = make_float3(hh,      -p_hsize, p_z-notch_depth);
    blocks[2].init(block3_points, iron, Fe_x);
}

Matrix* init_matrix(Settings* settings) {
    Matrix* matrix;
    cudaMallocManaged(&matrix, 1*sizeof(Matrix));

    // init matrix
    float matrix_width = settings->det_size;
    int matrix_width_px = settings->det_resolution;
    int matrix_height_px = settings->det_resolution;
    matrix->init(matrix_width_px, matrix_height_px, matrix_width/matrix_width_px, -90.0);

    return matrix;
}

void store_output(Matrix* matrix) {
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
}


float** xray_image(TubeType tube_type, float volatage, float power, float det_resolution, float det_size, float det_exposure, PartType part_type, float hole_size, float p_thicc) {
    printf("XI INIT\n");
    
    // settings
    Settings *settings;
    cudaMallocManaged(&settings, 1*sizeof(Settings));
    try {
        settings->init(tube_type, volatage, power, det_resolution, det_size, det_exposure);
    } catch (...) {
        std::cerr << "Error during settings init.\n";
        return nullptr;
    }

    printf("SETTINGS SET\n");

    // x-ray source
    float3 source = make_float3(0.0, 0.0, 0.0);

    // blocks representing 3d objects
    std::vector<Block> blocks_vec;
    Block* blocks;
    if(part_type == PT_bubble) 
        plate_with_holl(blocks_vec, settings, hole_size, p_thicc);
    else
        plate_with_notch(blocks_vec, settings, 1., hole_size, p_thicc);
    cudaMallocManaged(&blocks, blocks_vec.size()*sizeof(Block));
    std::copy(blocks_vec.begin(), blocks_vec.end(), blocks);

    printf("BLOCKS DATA LOADED\n");

    // sensor matrix
    Matrix* matrix = init_matrix(settings);

    printf("MATRIX SET\n");

    // start x-ray image calculation
    int threads_size = 32;
    dim3 threadsPerBlock(threads_size, threads_size);
    int blocks_width = ceil((float)matrix->width/threads_size);
    int blocks_height = ceil((float)matrix->height/threads_size);
    dim3 blocksShape(blocks_width, blocks_height);

    printf("PREPAIRING cuRAND\n");

    // init cuRAND for noise calculation
    curandState *global_state;
    cudaMallocManaged(&global_state, blocks_width*blocks_height*threads_size*threads_size*sizeof(curandState));
    setup_curand<<<blocksShape, threadsPerBlock>>>(global_state);

    printf("STARTING KERNEL\n");

    // run kernel to calculate x-ray image
    xray_image_kernel<<<blocksShape, threadsPerBlock>>>(source, blocks, blocks_vec.size(), matrix, settings, global_state);

    // wait for all threads and blocks
    gpuErrchk( cudaDeviceSynchronize() );

    printf("STORING OUTPUT\n");

    // store matrix as output.txt
    store_output(matrix);

    // print errors
    gpuErrchk( cudaPeekAtLastError() );

    // free allocated memory
    // cudaFree(blocks);
    // cudaFree(block1_points);
    // cudaFree(block2_points);
    // cudaFree(matrix);

    cudaProfilerStop();

    printf("XI STOPPED\n");

    return matrix->image;
}