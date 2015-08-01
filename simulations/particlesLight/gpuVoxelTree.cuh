#ifndef _GPUVOXELTREE_CUH_
#define _GPUVOXELTREE_CUH_
#include "gpuVoxelTree.h"

void copyDataToConstantMemory(const unsigned int numberOfLevels,
                              const BoundingBox BB,
                              const std::vector<unsigned int> & numberOfCellsPerSide,
                              const float sizeOfVoxel,
                              const std::vector<void *> & pointersToLevelStatuses,
                              const std::vector<void *> & pointersToLevelDelimiters,
                              const unsigned int numberOfVoxelsPerSide);

void collideWithParticles(float *particlePos,
                          float *particleVel,
                          float  particleRadius,
                          unsigned int numParticles,
                          unsigned int *numClaimedForLevel,
                          float deltaTime); 

void getPointersToDeallocateFromGPU(const unsigned int numberOfLevels,
                                    std::vector<void *> * statusPointersToDeallocate, 
                                    std::vector<void *> * delimiterPointersToDeallocate);

void generateMarchingCubes(float *pos,
                           float *norm,
                           unsigned int *tri,
                           unsigned int *numVerts,
                           unsigned int *verticesInPosArray,
                           unsigned int numVoxelsToDraw,
                           unsigned int numMarchingCubes);

#endif
