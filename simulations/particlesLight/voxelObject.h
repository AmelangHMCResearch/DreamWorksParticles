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

#define MAX_ROCK_STRENGTH 1

class VoxelObject {
public:
	enum ObjectShape
    {
        VOXEL_CUBE,
        VOXEL_PLANE,
        VOXEL_SPHERE,
        VOXEL_GEOLOGY
    };

    VoxelObject(ObjectShape shape, float voxelSize, uint3 cubeSize, float3 origin);
    ~VoxelObject();

    void initObject(ObjectShape shape);

    void generateLandscapeStrength();
    void initShape(ObjectShape shape);

    float getVoxelSize() {
    	return _objectParams._voxelSize;
    }

    uint3 getCubeSize() {
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

    float* getVoxelStrength()
    {
        return _dev_voxelStrength;
    }

    float* getPosArray();
    float* getCpuPosArray();
    float* getVoxelStrengthFromGPU();

    void unbindPosArray();

    unsigned int createVBO(unsigned int size);




private:

	// CPU Data
    float _maxVoxelStrength; 
	float *_voxelStrength;
    float *_pos; 
    unsigned int _numActiveVoxels;

    // GPU Data
    float *_dev_voxelStrength;
    unsigned int   _posVBO;            // vertex buffer object for particle positions
    unsigned int   _colorVBO;          // vertex buffer object for col
    struct cudaGraphicsResource *_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	// Parameters
	ObjectParams _objectParams;
	
};

#endif
