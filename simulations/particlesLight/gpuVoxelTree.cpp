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

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif


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
    BoundingBox boundary = {{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}};
    _boundary = boundary;
    checkCudaErrors(cudaMemcpy(_dev_boundary, &boundary, 1*sizeof(BoundingBox), cudaMemcpyHostToDevice));

    // to hold the pointers that will be copied to the GPU data members
    std::vector<void *> pointersToStatusesOnGPU(_numberOfLevels);
    std::vector<void *> pointersToDelimitersOnGPU(_numberOfLevels);

    // allocate space for all levels of tree

    // first, create the space for the data at each level
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];
        checkCudaErrors(cudaMalloc((void **) &pointersToStatusesOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &pointersToDelimitersOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(unsigned int)));
    }

    // then, create the space and copy the pointers to that data to the GPU
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelStatuses, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelDelimiters, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelStatuses, &pointersToStatusesOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelDelimiters, &pointersToDelimitersOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));


    // set the top level of the tree to active to represent the cube.
    const unsigned int numberOfTopLevelEntries = _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0];
    const float topLevel[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
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
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];

        // make enough room for the data to be copied
        statuses[levelIndex].reserve(numberOfEntriesInLevel);

        // copy the data over to CPU
        checkCudaErrors(cudaMemcpy(&statuses[levelIndex][0], pointersToStatusesOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(float), cudaMemcpyDeviceToHost));
    }

    // printf("Status of first four entries in top level of tree are:\n");
    // printf("%5.8f %5.8f %5.8f %5.8f\n", statuses[0][0], statuses[0][1], statuses[0][2], statuses[0][3]);
    return statuses;
}



std::vector<std::vector<unsigned int> > VoxelTree::getDelimiters() {
    // copy over the pointers to the delimiter data
    std::vector<void *> pointersToDelimitersOnGPU(_numberOfLevels);
    checkCudaErrors(cudaMemcpy(&pointersToDelimitersOnGPU[0], _dev_pointersToLevelDelimiters, _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));

    // space to hold the delimiters on CPU
    std::vector<std::vector<unsigned int> > delimiters(_numberOfLevels);

    // copy over all the allocated data, whether or not it is valid
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];

        // make enough room for the data to be copied
        delimiters[levelIndex].reserve(numberOfEntriesInLevel);

        // copy the data over to CPU
        checkCudaErrors(cudaMemcpy(&delimiters[levelIndex][0], pointersToDelimitersOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    return delimiters;
}


void VoxelTree::debugDisplay() {
    if (_isInitialized) {
        std::vector<std::vector<float> > statuses = getStatuses();
        std::vector<std::vector<unsigned int> > delimiters = getDelimiters();
    
        const unsigned int numberOfTopLevelEntries = _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0];

        printf("First four entries in top level of tree are:\n");
        // status
        printf("Status:");
        for (unsigned int index = 0; index < numberOfTopLevelEntries; ++index) {
            printf(" %5.2f", statuses[0][index]);
        }
        printf("\n");
        // delimiter
        printf("Delimiter:");
        for (unsigned int index = 0; index < numberOfTopLevelEntries; ++index) {
            // printf(" %u", delimiters[0][index]);
        }
        printf("\n");

        // const unsigned int numberOfVoxels = voxelObject->getNumVoxels();
        // const float * voxelPositionArray = voxelObject->getCpuPosArray();
        // const float * voxelStrength = voxelObject->getVoxelStrengthFromGPU();
        // const float voxelSize = voxelObject->getVoxelSize();

        unsigned int currentNumberOfCellsPerSide = _numberOfCellsPerSideForLevel[0];
        unsigned int currentSquaredCellsPerSide  = currentNumberOfCellsPerSide * currentNumberOfCellsPerSide;
        float currentCellSize = (_boundary.upperBoundary[0] - _boundary.lowerBoundary[0]) / currentNumberOfCellsPerSide;
        printf("VoxelSize is %3.2f\n", currentCellSize);
        
        for (unsigned int cellIndex = 0;
            cellIndex < 8; ++cellIndex) {
            
            unsigned int xIndex = cellIndex % currentNumberOfCellsPerSide;
            unsigned int yIndex = (cellIndex % currentSquaredCellsPerSide) / currentNumberOfCellsPerSide;
            unsigned int zIndex = cellIndex / currentSquaredCellsPerSide;

            float xPos = _boundary.lowerBoundary[0] + (0.5 + xIndex)*currentCellSize;
            float yPos = _boundary.lowerBoundary[1] + (0.5 + yIndex)*currentCellSize;
            float zPos = _boundary.lowerBoundary[2] + (0.5 + zIndex)*currentCellSize;

            if (statuses[0][cellIndex] > 0.0f) {
                printf("Drawing cell at (%5.8f, %5.8f, %5.8f) of size %5.8f\n", xPos, yPos, zPos, currentCellSize);
                // save the matrix state
                glPushMatrix();
                // translate for this voxel
                glTranslatef(xPos, yPos, zPos);
                             
                // float* color = new float[3];
                // getColor(statuses[voxelIndex]/(float)MAX_ROCK_STRENGTH, color);
                float color[3] = {1.0, 0, 0};
                glColor3f(color[0], color[1], color[2]);
                // delete [] color;
                glutWireCube(currentCellSize);
                // reset the matrix state
                glPopMatrix();
            }
        }
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






