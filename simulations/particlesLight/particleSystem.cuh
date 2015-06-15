/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 #include "event_timer.h"

extern "C"
{
    void allocateArray(void **devPtr, int size);
    void freeArray(void *devPtr);

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    void setParameters(SimParams *hostParams);

    void integrateSystem(float *pos,
                         float *vel,
                         float *force,
                         float *posAfterLastSort,
                         float deltaTime,
                         uint numParticles,
                         bool posAfterLastSortIsValid,
                         bool *pointHasMovedMoreThanThreshold,
                         EventTimer* timer);

    void calcCellIndices(uint  *cellIndex,
                         uint  *particleIndex,
                         float *pos,
                         int    numParticles,
                         EventTimer* timer);

    void sortParticles(uint *cellIndex, 
                       uint *particleIndex, 
                       uint numParticles, 
                       EventTimer* timer);

    void sortParticlesOnce(uint *cellIndex, 
                       float *pos,
                       float *vel, 
                       uint numParticles, 
                       EventTimer* timer);

    void copyArrays(float *pos,
                    float *tempPos,
                    float *vel,
                    float *tempVel,
                    uint   numParticles,
                    EventTimer* timer);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     uint  *cellIndex,
                                     uint  *particleIndex,
                                     float *pos,
                                     float *tempPos,
                                     float *posAfterLastSort,
                                     float *vel,
                                     float *tempVel,
                                     bool  *posAfterLastSortIsValid,
                                     bool  *pointHasMovedMoreThanThreshold, 
                                     uint   numParticles,
                                     uint   numCells,
                                     EventTimer* timer);

    void findCellStart(uint  *cellStart,
                       uint  *cellEnd,
                       uint  *cellIndex,
                       float *pos,
                       float *oldPos,
                       uint   numParticles,
                       uint   numCells,
                       EventTimer* timer);

    void collide(float *pos,
                 float *vel,
                 float *force,
                 uint  *cellIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint  *numNeighbors,
                 uint   numParticles,
                 uint   numCells,
                 EventTimer* timer);

    bool checkForResort(bool *pointHasMovedMoreThanThreshold);

}
