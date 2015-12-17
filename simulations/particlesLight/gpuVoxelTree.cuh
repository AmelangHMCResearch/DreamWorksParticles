#ifndef _GPUVOXELTREE_CUH_
#define _GPUVOXELTREE_CUH_
#include "gpuVoxelTree.h"

void copyDataToConstantMemory(const unsigned int numberOfLevels,
                              const BoundingBox BB,
                              const std::vector<unsigned int> & numberOfCellsPerSide,
                              const float sizeOfVoxel,
                              const std::vector<void *> & pointersToLevelStatuses,
                              const std::vector<void *> & pointersToLevelDelimiters,
                              const std::vector<void *> & pointersToLevelUpDelimiters,
                              const unsigned int numberOfVoxelsPerSide);

void collideWithParticles(float *particlePos,
                          float *particleVel,
                          const float  particleRadius,
                          const unsigned int numParticles,
                          float **dev_statuses,
                          unsigned int **dev_upIndices,
                          unsigned int **dev_downIndices,
                          unsigned int *dev_numClaimedForLevel,
                          unsigned int *dev_numInactiveforLevel,
                          unsigned int *memAllocatedAtLevel,
                          const unsigned int *maxMemAtLevel,
                          const unsigned int *numberOfCellsPerSideAtLevel,
                          unsigned int numberOfLevels,
                          const float deltaTime,
                          const unsigned int timestepIndex,
                          bool usingCoarsen); 

void generateMarchingCubes(float *pos,
                           float *norm,
                           unsigned int *tri,
                           unsigned int *numVerts,
                           unsigned int *verticesInPosArray,
                           unsigned int numVoxelsToDraw,
                           unsigned int numMarchingCubes);

void createShape(const float *result,
                 const unsigned int numberOfResults,
                 unsigned int *numClaimedInArrayAtLevel,
                 unsigned int *addressOfErrorField);

#endif
