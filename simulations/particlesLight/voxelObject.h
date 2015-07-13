#ifndef __VOXEL_OBJECT_H__
#define __VOXEL_OBJECT_H__

#include <GL/glew.h>
#include "particles_kernel.cuh"
//#include "particleSystem.h"
//#include "particleSystem.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "vector_types.h"
#include "vector_functions.h"
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include "math_constants.h"


#include <cstdlib>
#include <cstdio>
#include <string>

class VoxelObject {
public:
	enum ObjectShape
    {
        VOXEL_CUBE,
        VOXEL_PLANE,
        VOXEL_SPHERE,
    };

    VoxelObject(ObjectShape shape, float voxelSize, unsigned int cubeSize, float3 origin);
    ~VoxelObject();

    void initObject(ObjectShape shape);
    void initShape(ObjectShape shape);

    float getVoxelSize() {
    	return _objectParams._voxelSize;
    }

    unsigned int getCubeSize() {
        return _objectParams._cubeSize;
    }

    float3 getOrigin() {
        return _objectParams._origin;
    }

    unsigned int getNumVoxels() {
        return _objectParams._numVoxels;
    }

    unsigned int getNumActive() {
        return _numActiveVoxels;
    }

    unsigned int getCurrentReadBuffer() const
    {
        return _posVBO;
    }

    unsigned int getColorBuffer()       const
    {
        return _colorVBO;
    }

    int* getVoxelStrength()
    {
        return _dev_voxelStrength;
    }

    float* getPosArray();
    float* getCpuPosArray();
    int* getVoxelStrengthFromGPU();

    void unbindPosArray();

    unsigned int createVBO(unsigned int size);




private:

	// CPU Data
    int _maxVoxelStrength; 
	int *_voxelStrength;
    float *_pos; 
    unsigned int _numActiveVoxels;

    // GPU Data
    int *_dev_voxelStrength;
    unsigned int   _posVBO;            // vertex buffer object for particle positions
    unsigned int   _colorVBO;          // vertex buffer object for col
    struct cudaGraphicsResource *_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	// Parameters
	ObjectParams _objectParams;
	
};

#endif
