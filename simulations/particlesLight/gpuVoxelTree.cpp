/* gpuVoxelTree.cpp
 *
 * Author: Zakkai Davidson - zdavidson@hmc.edu
 * Date: 29 Jun 2015
 * 
 * Purpose: The class included in this file serves to represent a voxel based
 *          3D model on the GPU similar to that of OpenVDB. The data structure
 *          consists of tiers, each of which splits up the active space of
 *          interest into different sized cells. The lowest level contains the
 *          actual voxel data. The design of this data structure is simply to
 *          answer the question "is particle x inside a voxel?", and thus it is
 *          meant to be used primarily with particle simulations on the GPU.
 *
 */ 

#include "gpuVoxelTree.h"
#include "gpuVoxelTree.cuh"

#include <cuda_runtime.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h


// takes in a vector that determines the branching of the tree
VoxelTree::VoxelTree(std::vector<unsigned int> numberOfCellsPerSideForLevel, float voxelSize) :
    _isInitialized(false),
    _voxelSize(voxelSize)
{   
    _numberOfCellsPerSideForLevel = numberOfCellsPerSideForLevel;
    _numberOfLevels = numberOfCellsPerSideForLevel.size();

    uint lastLevel = _numberOfLevels - 1;
    _numVoxels = _numberOfCellsPerSideForLevel[lastLevel] * _numberOfCellsPerSideForLevel[lastLevel] * _numberOfCellsPerSideForLevel[lastLevel]; 

   // rest of setup is done in initializeTree()
}

VoxelTree::~VoxelTree()
{   
    // only clear data if it has been allocated
    if (_isInitialized) {
        std::vector<void *> statusPointersToDeallocateOnGPU(_numberOfLevels);
        std::vector<void *> delimeterPointersToDeallocateOnGPU(_numberOfLevels);
        getPointersToDeallocateFromGPU(statusPointersToDeallocateOnGPU, 
                                       delimeterPointersToDeallocateOnGPU,
                                       _numberOfLevels);
        for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
            checkCudaErrors(cudaFree(statusPointersToDeallocateOnGPU[levelIndex]));            
            checkCudaErrors(cudaFree(delimeterPointersToDeallocateOnGPU[levelIndex]));            
        }
    }


}


// NOTE:
/*
    This currently treats the first entry of the numberOfCellsPerSideForLevel array
    as the split for the bounding box of the entire object. This implicitly requires
    variable sized voxels because the bounding box determines determines the size
    of the smallest cell. This will likely cause errors for anything that isn't
    a cube and should be improved.
*/

        BoundingBox _boundary;  
        float *_pointersToLevelStatuses; 
        unsigned int *_pointersToLevelDelimiters;   
void VoxelTree::initializeTree()
{   
    // TODO: pass in VDB (?)
    // For now, we will just initialize a fixed size cube with arbitrarily-sized voxels

    // copy the bounding box to the GPU
    _boundary.lowerBoundary = make_float3(-1.0, -1.0, -1.0);
    _boundary.upperBoundary = make_float3(1.0, 1.0, 1.0);

    // to hold the pointers that will be copied to the GPU data members
    std::vector<void *> pointersToStatusesOnGPU(_numberOfLevels);
    std::vector<void *> pointersToDelimitersOnGPU(_numberOfLevels);

    // allocate space for all levels of tree

    // first, create the space for the data at each level
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];
        checkCudaErrors(cudaMalloc((void **) &pointersToStatusesOnGPU[0], numberOfEntriesInLevel*sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &pointersToDelimitersOnGPU[0], numberOfEntriesInLevel*sizeof(unsigned int)));
    }

    // set the top level of the tree to active to represent the cube.
    const unsigned int numberOfTopLevelEntries = _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0];
    checkCudaErrors(cudaMemset((pointersToStatusesOnGPU[0]), (unsigned int) 1, numberOfTopLevelEntries * sizeof(unsigned int)));

    copyDataToConstantMemory(_numberOfLevels,
                            _boundary,
                            _numberOfCellsPerSideForLevel,
                            _voxelSize,
                            pointersToStatusesOnGPU,
                            pointersToDelimitersOnGPU);

    _isInitialized = true;
}

void VoxelTree::runCollisions(float *particlePos, 
                              float *particleVel, 
                              float  particleRadius,
                              unsigned int numParticles)
{
    collideWithParticles(particlePos,
                         particleVel,
                         particleRadius,
                         numParticles,
                         _numVoxels); 
}

// ***************
// *** TESTING ***
// ***************

void VoxelTree::test()
{
    unsigned int blah[4] = {2, 2, 2, 2};
    std::vector<unsigned int> cellsPerSide(blah, blah + 4 * sizeof(unsigned int));
    float voxelSize = 1.0 / 128.0; 

    VoxelTree tree(cellsPerSide, voxelSize);
}






