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
    checkCudaErrors(cudaMalloc((void **) &_dev_numberOfCellsPerSideForLevel, _numberOfLevels*sizeof(unsigned int)));


    // set values on GPU to given configuration
    checkCudaErrors(cudaMemcpy(_dev_numberOfLevels, &_numberOfLevels, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_numberOfCellsPerSideForLevel, &numberOfCellsPerSideForLevel[0], 
                               _numberOfLevels*sizeof(unsigned int), cudaMemcpyHostToDevice));
}

VoxelTree::~VoxelTree()
{   
    // only clear data if it has been allocated
    if (_isInitialized) {
        std::vector<void *> statusPointersToDeallocateOnGPU(_numberOfLevels);
        std::vector<void *> delimeterPointersToDeallocateOnGPU(_numberOfLevels);
        checkCudaErrors(cudaMemcpy(&statusPointersToDeallocateOnGPU[0], _dev_pointersToLevelStatuses,
                                   _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&delimeterPointersToDeallocateOnGPU[0], _dev_pointersToLevelDelimiters,
                                   _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));
        for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
            checkCudaErrors(cudaFree(statusPointersToDeallocateOnGPU[levelIndex]));            
            checkCudaErrors(cudaFree(delimeterPointersToDeallocateOnGPU[levelIndex]));            
        }

        checkCudaErrors(cudaFree(_dev_boundary));
        checkCudaErrors(cudaFree(_dev_pointersToLevelStatuses));
        checkCudaErrors(cudaFree(_dev_pointersToLevelDelimiters));
        // checkCudaErrors(cudaFree(_dev_voxels));
    }

    // always free configuration data
    checkCudaErrors(cudaFree(_dev_numberOfLevels));
    checkCudaErrors(cudaFree(_dev_numberOfCellsPerSideForLevel));
}


// NOTE:
/*
    This currently treats the first entry of the numberOfCellsPerSideForLevel array
    as the split for the bounding box of the entire object. This implicitly requires
    variable sized voxels because the bounding box determines determines the size
    of the smallest cell. This will likely cause errors for anything that isn't
    a cube and should be improved.
*/
void VoxelTree::initializeTree()
{   
    // TODO: pass in VDB (?)
    // For now, we will just initialize a fixed size cube with arbitrarily-sized voxels

    // make the space for the bounding box and set it
    checkCudaErrors(cudaMalloc((void **) &_dev_boundary, 1*sizeof(BoundingBox)));

    // copy the bounding box to the GPU
    BoundingBox boundary = {{-1.0, -1.0}, {1.0, 1.0}};
    checkCudaErrors(cudaMemcpy(_dev_boundary, &boundary, 1*sizeof(BoundingBox), cudaMemcpyHostToDevice));

    // to hold the pointers that will be copied to the GPU data members
    std::vector<void *> pointersToStatusesOnGPU(_numberOfLevels);
    std::vector<void *> pointersToDelimitersOnGPU(_numberOfLevels);

    // allocate space for all levels of tree

    // first, create the space for the data at each level
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];
        checkCudaErrors(cudaMalloc((void **) &pointersToStatusesOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &pointersToDelimitersOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(unsigned int)));
    }

    // then, create the space and copy the pointers to that data to the GPU
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelStatuses, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelDelimiters, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelStatuses, &pointersToStatusesOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelDelimiters, &pointersToDelimitersOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));


    // set the top level of the tree to active to represent the cube.
    const unsigned int numberOfTopLevelEntries = _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0];
    const float topLevel[4] = {1.0, 1.0, 1.0, 1.0};
    checkCudaErrors(cudaMemcpy(pointersToStatusesOnGPU[0], topLevel, numberOfTopLevelEntries*sizeof(float), cudaMemcpyHostToDevice));

    _isInitialized = true;
}


std::vector<std::vector<float> > VoxelTree::getStatuses() {
    // copy over the pointers to the status data
    std::vector<void *> pointersToStatusesOnGPU(_numberOfLevels);
    checkCudaErrors(cudaMemcpy(&pointersToStatusesOnGPU[0], _dev_pointersToLevelStatuses, _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));

    // space to hold the statuses on CPU
    std::vector<std::vector<float> > statuses(_numberOfLevels);

    // copy over all the allocated data, whether or not it is valid
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];

        // make enough room for the data to be copied
        statuses[levelIndex].reserve(numberOfEntriesInLevel);

        // copy the data over to CPU
        checkCudaErrors(cudaMemcpy(&statuses[levelIndex][0], pointersToStatusesOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(float), cudaMemcpyDeviceToHost));
    }

    // printf("Status of first four entries in top level of tree are:\n");
    // printf("%5.8f %5.8f %5.8f %5.8f\n", statuses[0][0], statuses[0][1], statuses[0][2], statuses[0][3]);
    return statuses;
}


void VoxelTree::debugDisplay() {
    if (_isInitialized) {
        std::vector<std::vector<float> > statuses = getStatuses();
        // printf("Status of first four entries in top level of tree are:\n");
        // printf("%5.8f %5.8f %5.8f %5.8f\n", statuses[0][0], statuses[0][1], statuses[0][2], statuses[0][3]);
        // const unsigned int numberOfVoxels = voxelObject->getNumVoxels();
        // const float * voxelPositionArray = voxelObject->getCpuPosArray();
        // const float * voxelStrength = voxelObject->getVoxelStrengthFromGPU();
        // const float voxelSize = voxelObject->getVoxelSize();
        // for (unsigned int voxelIndex = 0;
        //     voxelIndex < numberOfVoxels; ++voxelIndex) {
        //     if (voxelStrength[voxelIndex] > 0.0f) {
        //         // save the matrix state
        //         glPushMatrix();
        //         // translate for this voxel
        //         glTranslatef(voxelPositionArray[voxelIndex * 4 + 0],
        //                      voxelPositionArray[voxelIndex * 4 + 1],
        //                      voxelPositionArray[voxelIndex * 4 + 2]);
        //         float* color = new float[3];
        //         getColor(voxelStrength[voxelIndex]/(float)MAX_ROCK_STRENGTH, color);
        //         glColor3f(color[0], color[1], color[2]);
        //         delete [] color;
        //         glutSolidCube(voxelSize);
        //         // reset the matrix state
        //         glPopMatrix();
        //     }
        // }
    }
}


// ***************
// *** TESTING ***
// ***************

void VoxelTree::test()
{
    unsigned int blah[4] = {2, 2, 2, 2};
    std::vector<unsigned int> cellsPerSide(blah, blah + sizeof(blah) / sizeof(blah[0]));
    // printf("size of cellsPerSide is %lu\n", cellsPerSide.size());

    VoxelTree tree(cellsPerSide);
    tree.initializeTree();
    tree.debugDisplay();
}






