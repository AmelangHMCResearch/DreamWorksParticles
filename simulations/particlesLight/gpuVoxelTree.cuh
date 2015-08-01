#ifndef _GPUVOXELTREE_CUH_
#define _GPUVOXELTREE_CUH_
#include "gpuVoxelTree.h"

void copyDataToConstantMemory(unsigned int numberOfLevels,
                             BoundingBox BB, 
                             std::vector<unsigned int> numberOfCellsPerSide,
                             float sizeOfVoxel,
                             std::vector<void *> pointersToLevelStatuses,
                             std::vector<void *> pointersToLevelDelimiters,
                             unsigned int numberOfVoxelsPerSide);

void collideWithParticles(float *particlePos,
                          float *particleVel,
                          float  particleRadius,
                          unsigned int numParticles,
                          float **pointersToStatuses,
                          unsigned int **pointersToDelimiters,
                          unsigned int *numClaimedForLevel,
                          float deltaTime); 

void getPointersToDeallocateFromGPU(std::vector<void *> statusPointersToDeallocate, 
                                    std::vector<void *> delimiterPointersToDeallocate,
                                    uint numLevels);

void generateMarchingCubes(float *pos,
                           float *norm,
                           float **pointersToStatuses,
                           unsigned int **pointersToDelimiters,
                           unsigned int *tri,
                           unsigned int *numVerts,
                           unsigned int *verticesInPosArray,
                           unsigned int numVoxelsToDraw,
                           unsigned int numMarchingCubes);

#endif