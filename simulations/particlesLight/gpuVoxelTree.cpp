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

#include <cuda_runtime.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h


// takes in a vector that determines the branching of the tree
VoxelTree::VoxelTree(std::vector<unsigned int> numberOfCellsPerSideForLevel) :
    _isInitialized(false)
{   
    _numberOfCellsPerSideForLevel = numberOfCellsPerSideForLevel;
    _numberOfLevels = numberOfCellsPerSideForLevel.size();

    // allocate space for the configuration values
    checkCudaErrors(cudaMalloc((void **) &_dev_numberOfLevels, 1*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void **) &_dev_numberOfCellsPerSideForLevel, numberOfLevels*sizeof(unsigned int)));


    // set values on GPU to given configuration
    checkCudaErrors(cudaMemcpy(_dev_numberOfLevels, &_numberOfLevels, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_numberOfCellsPerSideForLevel, &numberOfCellsPerSideForLevel[0], 
                               numberOfLevels*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

VoxelTree::~VoxelTree()
{   
    // only clear data if it has been allocated
    if (_isInitialized) {
        std::vector<unsigned int> statusPointersToDeallocateOnGPU(_numberOfLevels);
        std::vector<unsigned int> delimeterPointersToDeallocateOnGPU(_numberOfLevels);
        checkCudaErrors(cudaMemcpy(&statusPointersToDeallocateOnGPU[0], _dev_pointersToLevelStatuses,
                                   _numberOfLevels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&delimeterPointersToDeallocateOnGPU[0], _dev_pointersToLevelDelimiters,
                                   _numberOfLevels*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
            checkCudaErrors(cudaFree(statusPointersToDeallocateOnGPU[levelIndex]));            
            checkCudaErrors(cudaFree(delimeterPointersToDeallocateOnGPU[levelIndex]));            
        }

        checkCudaErrors(cudaFree(_dev_boundary));
        checkCudaErrors(cudaFree(_dev_pointersToLevelStatuses));
        checkCudaErrors(cudaFree(_dev_pointersToLevelDelimiters));
        checkCudaErrors(cudaFree(_dev_voxels));
    }

    // always free configuration data
    checkCudaErrors(cudaFree(_dev_numberOfLevels));
    checkCudaErrors(cudaFree(_dev_numberOfCellsPerSideForLevel));


}

void VoxelTree::initializeTree()
{   
    // TODO
    // For now, we will just initialize a fixed size cube with arbitrarily-sized voxels

    // make the space for the bounding box and set it
    checkCudaErrors(cudaMalloc((void **) &_dev_boundary, 1*sizeof(BoundingBox)));

    // copy the bounding box to the GPU
    BoundingBox boundary = {{-1.0, -1.0}, {1.0, 1.0}};
    checkCudaErrors(cudaMemcpy(_dev_boundary, &boundary, 1*sizeof(BoundingBox), cudaMemcpyHostToDevice));

    // allocate space for all levels of tree
    unsigned int numberOfEntriesInLevel = 4; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {

    }




    // set the top level of the tree to active to represent the cube.
    std::vector<unsigned int> dummyLevelData = {1, 1, 1, 1};
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelStatuses)
}

