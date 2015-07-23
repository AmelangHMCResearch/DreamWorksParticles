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
        VOXEL_GEOLOGY,
        VOXEL_FROM_FILE
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

    unsigned int getNumVoxelsToDraw() {
        return _objectParams._maxVoxelsToDraw;
    }

    unsigned int getNumMarchingCubes() {
        return (_objectParams._cubeSize.x + 1) * (_objectParams._cubeSize.y + 1) * (_objectParams._cubeSize.z + 1);
    }

    unsigned int getCPUNumVoxelsDrawn() {
        uint numDrawn;
        cudaMemcpy(&numDrawn, _dev_verticesInPosArray, sizeof(uint), cudaMemcpyDeviceToHost);
        return numDrawn;
    }

    unsigned int getNumActive() {
        return _numActiveVoxels;
    }

    unsigned int getCurrentReadBuffer() const
    {
        return _posVBO;
    }

    unsigned int getNormalBuffer() const
    {
        return _normVBO;
    }

    uint* getTriTable()
    {
        return _dev_triTable;
    }

    uint* getNumVertsTable()
    {
        return _dev_numVertsTable; 
    }

    float* getVoxelStrength()
    {
        return _dev_voxelStrength;
    }

    uint* getVerticesInPosArray()
    {
        return _dev_verticesInPosArray;
    }

    uint getVoxelIndex(uint3 gridPos);
    bool voxelIsInBounds(uint3 gridPos);

    float* getPosArray();
    float* getNormalArray();
    float* getCpuPosArray();
    float* getVoxelStrengthFromGPU();

    void unbindPosArray();
    void unbindNormalArray();

    unsigned int createVBO(unsigned int size);




private:

	// CPU Data
    float _maxVoxelStrength; 
	float *_voxelStrength;
    float *_pos; 
    unsigned int _numActiveVoxels;

    // GPU Data
    float *_dev_voxelStrength;
    uint *_dev_verticesInPosArray; 
    uint *_dev_triTable;
    uint *_dev_numVertsTable;
    unsigned int   _posVBO;            // vertex buffer object for particle positions
    unsigned int   _normVBO;
    struct cudaGraphicsResource *_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *_cuda_normvbo_resource;

	// Parameters
	ObjectParams _objectParams;
	
};

#endif
