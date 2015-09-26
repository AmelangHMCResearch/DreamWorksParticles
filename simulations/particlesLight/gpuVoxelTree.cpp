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

#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h

#include <math.h>

#include "gpuVoxelTree.h"
#include "gpuVoxelTree.cuh"
#include "tables.h"


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

   // rest of setup is done in initializeTree()
}

VoxelTree::~VoxelTree()
{   
    // only clear data if it has been allocated
    if (_isInitialized) {
        std::vector<void *> statusPointersToDeallocateOnGPU(_numberOfLevels);
        std::vector<void *> downDelimeterPointersToDeallocateOnGPU(_numberOfLevels);
        std::vector<void *> upDelimeterPointersToDeallocateOnGPU(_numberOfLevels);

        checkCudaErrors(cudaMemcpy(&statusPointersToDeallocateOnGPU[0], _dev_pointersToLevelStatuses,
                                   _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&upDelimeterPointersToDeallocateOnGPU[0], _dev_pointersToLevelDownDelimiters,
                                   _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&downDelimeterPointersToDeallocateOnGPU[0], _dev_pointersToLevelUpDelimiters,
                                   _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));

        for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
            checkCudaErrors(cudaFree(statusPointersToDeallocateOnGPU[levelIndex]));         
            checkCudaErrors(cudaFree(upDelimeterPointersToDeallocateOnGPU[levelIndex])); 
            checkCudaErrors(cudaFree(downDelimeterPointersToDeallocateOnGPU[levelIndex]));            
        }

        checkCudaErrors(cudaFree(_dev_boundary));
        checkCudaErrors(cudaFree(_dev_pointersToLevelStatuses));
        checkCudaErrors(cudaFree(_dev_pointersToLevelDownDelimiters));
        // checkCudaErrors(cudaFree(_dev_voxels));
    }

    // always free configuration data
    checkCudaErrors(cudaFree(_dev_numberOfLevels));
    checkCudaErrors(cudaFree(_dev_numberOfCellsPerSideForLevel));

    checkCudaErrors(cudaFree(_dev_triTable));
    checkCudaErrors(cudaFree(_dev_numVertsTable));
    
    checkCudaErrors(cudaGraphicsUnregisterResource(_cuda_posvbo_resource));
    glDeleteBuffers(1, (const GLuint *)&_posVBO);

    checkCudaErrors(cudaGraphicsUnregisterResource(_cuda_normvbo_resource));
    glDeleteBuffers(1, (const GLuint *)&_normVBO);
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
    // printf("Starting to initialize tree!!!!\n");
    // TODO: pass in VDB (?)
    // For now, we will just initialize a fixed size cube with arbitrarily-sized voxels

    // copy the bounding box to the GPU
    _boundary.lowerBoundary = make_float3(-1.0, -1.0, -1.0);
    _boundary.upperBoundary = make_float3(1.0, 1.0, 1.0);

    checkCudaErrors(cudaMalloc((void **) &_dev_boundary, 1*sizeof(BoundingBox)));

    // copy the bounding box to the GPU
    checkCudaErrors(cudaMemcpy(_dev_boundary, &_boundary, 1*sizeof(BoundingBox), cudaMemcpyHostToDevice));

    // to hold the pointers that will be copied to the GPU data members
    std::vector<void *> pointersToStatusesOnGPU(_numberOfLevels);
    std::vector<void *> pointersToDownDelimitersOnGPU(_numberOfLevels);
    std::vector<void *> pointersToUpDelimitersOnGPU(_numberOfLevels);

    // allocate space for all levels of tree

    // first, create the space for the data at each level
    unsigned int numberOfEntriesInLevel = 1; // based on number of cells per side TODO
    unsigned int numberOfCellsPerSideInLevel = 1; 
    for (unsigned int levelIndex = 0; levelIndex < _numberOfLevels; ++levelIndex) {
        // First handle upchunk array - create vector holding entries, and copy them over. 
        // number of chunks in level = number of entries in prev level
        checkCudaErrors(cudaMalloc((void **) &pointersToUpDelimitersOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(unsigned int)));

        numberOfEntriesInLevel *= _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex] * _numberOfCellsPerSideForLevel[levelIndex];
        numberOfCellsPerSideInLevel *= _numberOfCellsPerSideForLevel[levelIndex];
        printf("Initializing %d elements on level %d\n", numberOfEntriesInLevel, levelIndex);
        checkCudaErrors(cudaMalloc((void **) &pointersToStatusesOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &pointersToDownDelimitersOnGPU[levelIndex], numberOfEntriesInLevel*sizeof(unsigned int)));
    }
    _voxelSize = (_boundary.upperBoundary.x - _boundary.lowerBoundary.x) / numberOfCellsPerSideInLevel;
    _numMarchingCubes = (numberOfCellsPerSideInLevel + 1) * (numberOfCellsPerSideInLevel + 1) * (numberOfCellsPerSideInLevel + 1);

    // then, create the space and copy the pointers to that data to the GPU
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelStatuses, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelDownDelimiters, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMalloc((void **) &_dev_pointersToLevelUpDelimiters, _numberOfLevels*sizeof(void *)));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelStatuses, &pointersToStatusesOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelDownDelimiters, &pointersToDownDelimitersOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_dev_pointersToLevelUpDelimiters, &pointersToUpDelimitersOnGPU[0], _numberOfLevels*sizeof(void *), cudaMemcpyHostToDevice));

    // set the top level of the tree to active to represent the cube.
    const unsigned int numberOfTopLevelEntries = _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0] * _numberOfCellsPerSideForLevel[0];    
    std::vector<float> topLevel(numberOfTopLevelEntries, 0.0001);
    // decompose one cell of the cube
    //topLevel[numberOfTopLevelEntries - 1] = STATUS_FLAG_DIG_DEEPER;
    checkCudaErrors(cudaMemcpy(pointersToStatusesOnGPU[0], &topLevel[0], numberOfTopLevelEntries*sizeof(float), cudaMemcpyHostToDevice));

    // set the delimiters for the top level to point toward test cell
    std::vector<unsigned int> topLevelDelimiters(numberOfTopLevelEntries, INVALID_CHUNK_NUMBER);
    //topLevelDelimiters[numberOfTopLevelEntries - 1] = numberOfTopLevelEntries - 1;
    checkCudaErrors(cudaMemcpy(pointersToDownDelimitersOnGPU[0], &topLevelDelimiters[0], numberOfTopLevelEntries*sizeof(unsigned int), cudaMemcpyHostToDevice));
    /*// Then, set all up delimiters to invalid to begin with
    unsigned int numberOfChunksInLevel = 1; 
    for (int i = 0; i < _numberOfLevels; ++i) {
        std::vector<unsigned int> delimiters(numberOfChunksInLevel, INVALID_CHUNK_NUMBER);
        checkCudaErrors(cudaMemcpy(pointersToUpDelimitersOnGPU[i], &delimiters[0], delimiters.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        numberOfChunksInLevel *= _numberOfCellsPerSideForLevel[i] * _numberOfCellsPerSideForLevel[i] * _numberOfCellsPerSideForLevel[i];
    }*/

    std::vector<unsigned int> numClaimedForLevel(_numberOfLevels, 0); 

    checkCudaErrors(cudaMalloc((void**) &_dev_numClaimedForLevel, _numberOfLevels * sizeof(unsigned int))); 
    checkCudaErrors(cudaMemcpy(_dev_numClaimedForLevel, &numClaimedForLevel[0], _numberOfLevels * sizeof(unsigned int), cudaMemcpyHostToDevice));

    copyDataToConstantMemory(_numberOfLevels,
                             _boundary,
                             _numberOfCellsPerSideForLevel,
                             _voxelSize,
                             pointersToStatusesOnGPU,
                             pointersToDownDelimitersOnGPU,
                             pointersToUpDelimitersOnGPU,
                             numberOfCellsPerSideInLevel);

    // More rendering stuff: 

    checkCudaErrors(cudaMalloc((void **) &_dev_verticesInPosArray, sizeof(uint)));

    // Allocate lookup tables for marching cubes on the GPU
    checkCudaErrors(cudaMalloc((void **) &_dev_triTable, sizeof(uint) * 256 * 16));
    cudaMemcpy(_dev_triTable, triTable, sizeof(uint) * 256 * 16, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMalloc((void **) &_dev_numVertsTable, sizeof(uint) * 256));
    cudaMemcpy(_dev_numVertsTable, numVertsTable, sizeof(uint) * 256, cudaMemcpyHostToDevice);

    // Create the VBO
    _numVoxelsToDraw = std::min(_numMarchingCubes, (uint) 128 * 128 * 128);
    uint bufferSize = _numVoxelsToDraw * 4 * 15 * sizeof(float); 
    glGenBuffers(1, &_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cuda_posvbo_resource, _posVBO,
                                                     cudaGraphicsMapFlagsNone));

    glGenBuffers(1, &_normVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _normVBO);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cuda_normvbo_resource, _normVBO,
                                                     cudaGraphicsMapFlagsNone));

    _isInitialized = true;
    printf("Completed initialization\n");
}


void VoxelTree::initializeShape() {
    FILE *fp = fopen("BostonTeapot.raw", "rb");
    if (fp == NULL) {
        printf("Problem opening file\n");
        exit(1);
    }

    /*unsigned short dataSize[3];
    int result = fread((void*) &dataSize[0],sizeof(unsigned short), 3,fp);
    if (result == 0) {
        printf("Problem reading file 2\n");
        exit(1); 
    }*/
    const unsigned short dataSize[3] = {256, 256, 178};
    const unsigned int xWidth = dataSize[0];
    const unsigned int yWidth = dataSize[1];
    const unsigned int zWidth = dataSize[2];
    const unsigned int numberOfVoxelsInFile = dataSize[0] * dataSize[1] * dataSize[2];

    std::vector<unsigned char> dataFromFile(numberOfVoxelsInFile);
    unsigned int result = fread(&dataFromFile[0], sizeof(char), numberOfVoxelsInFile, fp);
    if (result == 0 && (feof(fp) || ferror(fp))) {
       printf("Problem reading file. Exiting...\n");
       exit(1);
    }
    printf("Grabbed %d voxels of %d from the file.\n", result, numberOfVoxelsInFile);
    

    // convert the voxels to an array of positions
    const unsigned int strengthThreshold = 60;
    const float voxelSize = 2.0 / 256.0;
    const float initialHardness = 10;
    BoundingBox overallBoundary = _boundary; 


    std::vector<float> shapeToRepair;

    for (unsigned int zIndex = 0; zIndex < zWidth; ++zIndex) {
        for (unsigned int yIndex = 0; yIndex < xWidth; ++yIndex) {
            for (unsigned int xIndex = 0; xIndex < yWidth; ++xIndex) {
                const unsigned int cellIndex = (xWidth * yWidth * zIndex) + (xWidth * yIndex) + xIndex;
                const unsigned char strength = dataFromFile[cellIndex];
                if (strength > strengthThreshold) {
                    const float xPos = overallBoundary.lowerBoundary.x + xIndex * voxelSize;
                    const float yPos = -1.0 * (overallBoundary.lowerBoundary.y + yIndex * voxelSize);
                    const float zPos = overallBoundary.lowerBoundary.z + zIndex * voxelSize;
                    // const float4 voxelPos = make_float4(xPos, yPos, zPos, -initialHardness);
                    shapeToRepair.push_back(xPos);
                    shapeToRepair.push_back(yPos);
                    shapeToRepair.push_back(zPos);
                    shapeToRepair.push_back(initialHardness);
                }
            }
        }
    }

    // make repair do all the work for us!
    float* dev_shapeToRepair;
    const unsigned int numberOfFloatsToSend = shapeToRepair.size();
    checkCudaErrors(cudaMalloc((void**) &dev_shapeToRepair, numberOfFloatsToSend*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dev_shapeToRepair, &shapeToRepair[0], numberOfFloatsToSend*sizeof(float), cudaMemcpyHostToDevice));

    unsigned int *dev_numErrors; 
    checkCudaErrors(cudaMalloc((void **)&dev_numErrors, sizeof(unsigned int))); 
    

    const unsigned int numberOfVoxelToSend = numberOfFloatsToSend / 4;
    const unsigned int numberOfSendingRounds = 1;
    const unsigned int numberOfVoxelToSendPerBatch = numberOfVoxelToSend / numberOfSendingRounds;

    printf("Sending %u voxels in %u rounds for total of %u of %u\n", numberOfVoxelToSendPerBatch, numberOfSendingRounds, numberOfSendingRounds*numberOfVoxelToSendPerBatch, numberOfVoxelToSend);

    for (unsigned int currentRound = 0; currentRound < numberOfSendingRounds; ++currentRound) {
        createShape((float *) dev_shapeToRepair + currentRound*numberOfVoxelToSendPerBatch,
                    numberOfVoxelToSendPerBatch,
                    _dev_numClaimedForLevel,
                    dev_numErrors);
    }

    checkCudaErrors(cudaFree(dev_shapeToRepair)); 
    checkCudaErrors(cudaFree(dev_numErrors)); 




    // // build the tree from the voxels
    // std::vector<std::vector<float> > statuses;
    // std::vector<std::vector<unsigned int> > delimiters;

    // for (unsigned int z = 0; z < dataSize[2]; z++)
    // {
    //     for (unsigned int y = 0; y < dataSize[1]; y++)
    //     {
    //         for (unsigned int x = 0; x < dataSize[0]; x++)
    //         {
    //             unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;
    //             if (x < dataSize[0] && y < dataSize[1] && z < dataSize[2]) {
    //                 unsigned short strength; 
    //                 int result = fread(&strength, sizeof(char), 1, fp);
    //                 if (result == 0 && (feof(fp) || ferror(fp))) {
    //                    printf("Problem reading file at index %d\n", i);
    //                    exit(1); 
    //                 }
    //                 if (strength < 60) {
    //                     _voxelStrength[i] = 0; 
    //                 } else {
    //                     _voxelStrength[i] = _maxVoxelStrength; 
    //                 }
                    
    //             } else {
    //                 _voxelStrength[i] = 0; 
    //             }
    //         }
    //     }
    // }
    fclose(fp); 
}



void VoxelTree::runCollisions(float *particlePos, 
                              float *particleVel, 
                              float  particleRadius,
                              float deltaTime,
                              unsigned int numParticles)
{
    collideWithParticles(particlePos,
                         particleVel,
                         particleRadius,
                         numParticles,
                         _dev_numClaimedForLevel,
                         _numberOfLevels,
                         deltaTime); 
}

void VoxelTree::renderVoxelTree(float modelView[16], float particleRadius)
{
    float *dPos;
    checkCudaErrors(cudaGraphicsMapResources(1, &_cuda_posvbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dPos, NULL,
                                                             _cuda_posvbo_resource));
    float *dNorm;
    checkCudaErrors(cudaGraphicsMapResources(1, &_cuda_normvbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dNorm, NULL,
                                                             _cuda_normvbo_resource));

    generateMarchingCubes(dPos,
                          dNorm,
                          _dev_triTable,
                          _dev_numVertsTable,
                          _dev_verticesInPosArray,
                          _numVoxelsToDraw,
                          _numMarchingCubes);
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cuda_posvbo_resource, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cuda_normvbo_resource, 0));

    // Set lighting and view
    float lightPos[] = { -3.0f, -15.0f, -3.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHTING);
    

    // Bind the buffers for openGL to use    
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, _normVBO);
    glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

    // Figure out how many vertices to draw, and use them for triangles
    uint num1 = _numVoxelsToDraw * 4 * 15;
    uint num2;
    cudaMemcpy(&num2, _dev_verticesInPosArray, sizeof(uint), cudaMemcpyDeviceToHost);
    uint numToDraw = std::min(num1, num2);
    glDrawArrays(GL_TRIANGLES, 0, numToDraw);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisable(GL_LIGHTING);

    glPopMatrix();
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
    checkCudaErrors(cudaMemcpy(&pointersToDelimitersOnGPU[0], _dev_pointersToLevelDownDelimiters, _numberOfLevels*sizeof(void *), cudaMemcpyDeviceToHost));

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

        //printf("First four entries in top level of tree are:\n");
        // status
        /*printf("Status:");
        for (unsigned int index = 0; index < numberOfTopLevelEntries; ++index) {
            printf(" %5.2f", statuses[1][index]);
        }
        printf("\n");
        // delimiter
        printf("Delimiter:");
        for (unsigned int index = 0; index < numberOfTopLevelEntries; ++index) {
            printf(" %u", delimiters[0][index]);
        }
        printf("\n");*/

        // make the first call to drawCell (called recursively to draw lower level cells)
        // TODO
        drawCell(statuses, delimiters, 0, 0, _boundary);   
    }
}



inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}


// create a color ramp
void findColor(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}


void VoxelTree::drawCell(std::vector<std::vector<float> > & statuses,
                         std::vector<std::vector<unsigned int> > & delimiters,
                         unsigned int delimiterForCurrentCell,
                         unsigned int currentLevel,
                         BoundingBox currentBoundary) {
    
    const unsigned int currentNumberOfCellsPerSide = _numberOfCellsPerSideForLevel[currentLevel];
    const unsigned int currentSquaredCellsPerSide  = currentNumberOfCellsPerSide * currentNumberOfCellsPerSide; 
    const float currentCellSize = (currentBoundary.upperBoundary.x - currentBoundary.lowerBoundary.x) / currentNumberOfCellsPerSide;
    // printf("VoxelSize is %3.2f\n", currentCellSize);

    const unsigned int numberOfEntriesInCell = currentNumberOfCellsPerSide * currentNumberOfCellsPerSide * currentNumberOfCellsPerSide;

    for (unsigned int cellIndex = 0; cellIndex < numberOfEntriesInCell; ++cellIndex) {
        
        unsigned int xIndex = cellIndex % currentNumberOfCellsPerSide;
        unsigned int yIndex = (cellIndex % currentSquaredCellsPerSide) / currentNumberOfCellsPerSide;
        unsigned int zIndex = cellIndex / currentSquaredCellsPerSide;

        float xPos = currentBoundary.lowerBoundary.x + (0.5 + xIndex)*currentCellSize;
        float yPos = currentBoundary.lowerBoundary.y + (0.5 + yIndex)*currentCellSize;
        float zPos = currentBoundary.lowerBoundary.z + (0.5 + zIndex)*currentCellSize;

        // actual index in status/delimiter arrays
        unsigned int actualCellIndex = cellIndex + delimiterForCurrentCell * numberOfEntriesInCell;

        float currentStatus = statuses[currentLevel][actualCellIndex];
        if (currentStatus != STATUS_FLAG_DIG_DEEPER && (currentLevel != _numberOfLevels - 1 || currentStatus > 0.0f)) {
            // printf("statuses[currentLevel][actualCellIndex] is %f\n", statuses[currentLevel][actualCellIndex]);
            // printf("Drawing cell at (%5.8f, %5.8f, %5.8f) of size %5.8f\n", xPos, yPos, zPos, currentCellSize);
            // save the matrix state
            glPushMatrix();
            // translate for this voxel
            glTranslatef(xPos, yPos, zPos);
                         
            float color[3];
            float t = currentLevel / (float) _numberOfLevels;
            findColor(t, color);
            // getColor(statuses[voxelIndex]/(float)MAX_ROCK_STRENGTH, color);
            // float color[3] = {1.0, 0, 0};
            glColor3f(color[0], color[1], color[2]);
            // delete [] color;
            if (currentStatus > 0.0f) {
                glutWireCube(currentCellSize);
            } else {
                glutSolidCube(currentCellSize);
            }
            // reset the matrix state
            glPopMatrix();
        } else if (currentStatus == STATUS_FLAG_DIG_DEEPER) {
            //printf("Need to dig deeper for cell %d on level %d\n", cellIndex, currentLevel);

            unsigned int delimiterForNextCell = delimiters[currentLevel][actualCellIndex];
            // printf("NextDelimiter is %d\n", delimiterForNextCell);
            unsigned int nextLevel = currentLevel + 1;
            float halfCellSize = 0.5 * currentCellSize;
            BoundingBox nextBoundary;
            nextBoundary.lowerBoundary = make_float3(xPos - halfCellSize, yPos - halfCellSize, zPos - halfCellSize);
            nextBoundary.upperBoundary = make_float3(xPos + halfCellSize, yPos + halfCellSize, zPos + halfCellSize);

            drawCell(statuses, delimiters, delimiterForNextCell, nextLevel, nextBoundary);
        } else { 
            if (currentLevel != _numberOfLevels - 1) {
                glPushMatrix();
                // translate for this voxel
                glTranslatef(xPos, yPos, zPos);
                             
                float color[3];
                float t = currentLevel / (float) _numberOfLevels;
                findColor(t, color);
                // getColor(statuses[voxelIndex]/(float)MAX_ROCK_STRENGTH, color);
                // float color[3] = {1.0, 0, 0};
                //glColor3f(1, 1, 1);
                // delete [] color;
                //glutSolidCube(currentCellSize);
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






