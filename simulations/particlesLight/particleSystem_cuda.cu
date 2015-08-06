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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"
#include "event_timer.h"

extern "C"
{
    void cudaGLInit(int argc, char **argv)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        findCudaGLDevice(argc, (const char **)argv);
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsNone));
    }

    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
                                                             *cuda_vbo_resource));
        return ptr;
    }

    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    void copyArrayFromDevice(void *host, const void *device,
                             struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource)
        {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource)
        {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void integrateSystem(float *pos,
                         float *vel,
                         float *force,
                         float *posAfterLastSort,
                         float deltaTime,
                         uint numParticles,
                         bool posAfterLastSortIsValid,
                         bool *pointHasMovedMoreThanThreshold,
                         uint *numParticlesToRemove,
                         EventTimer* timer)                 
    {
        if (numParticles == 0) {
          return;
        }
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        timer->startTimer(0, false);
        integrateSystemD<<< numBlocks, numThreads >>>((float4 *) pos,
                                                      (float4 *) vel,
                                                      (float4 *) force,
                                                      (float4 *) posAfterLastSort, 
                                                      deltaTime,
                                                      numParticles,
                                                      posAfterLastSortIsValid, 
                                                      pointHasMovedMoreThanThreshold,
                                                      numParticlesToRemove);
        timer->stopTimer(0, false);
        getLastCudaError("Kernel execution failed");
    }

    void calcCellIndices(uint  *cellIndex,
                         uint  *particleIndex,
                         float *pos,
                         float *vel,
                         int    numParticles,
                         EventTimer* timer)
    {
        if (numParticles == 0) {
          return;
        }
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        timer->startTimer(1, true);
        calcCellIndicesD<<< numBlocks, numThreads >>>(cellIndex,
                                               particleIndex,
                                               (float4 *) pos,
                                               (float4 *) vel,
                                               numParticles);
        timer->stopTimer(1, true);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }
    void sortParticles(uint *cellIndex, 
                       uint *particleIndex, 
                       uint numParticles, 
                       EventTimer* timer)
    {
        timer->startTimer(2, false);
        thrust::sort_by_key(thrust::device_ptr<uint>(cellIndex),
                            thrust::device_ptr<uint>(cellIndex + numParticles),
                            thrust::device_ptr<uint>(particleIndex));
        timer->stopTimer(2, false);
    }

    void copyArrays(float *pos,
                    float *tempPos,
                    float *vel,
                    float *tempVel,
                    uint   numParticles,
                    EventTimer* timer)
    {
        if (numParticles == 0) {
          return;
        }
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
#endif

        timer->startTimer(3, true);
        copyArraysD<<< numBlocks, numThreads>>>(
            (float4 *)pos,
            (float4 *)tempPos,
            (float4 *)vel,
            (float4 *)tempVel,
            numParticles);
        timer->stopTimer(3, true);
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(posTex));
        checkCudaErrors(cudaUnbindTexture(velTex));
#endif

    }

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
                                     EventTimer* timer)
    {
        if (numParticles == 0) {
          return;
        }
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

        uint smemSize = sizeof(uint)*(numThreads+1);

#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, tempPosTex, tempPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, tempVelTex, tempVel, numParticles*sizeof(float4)));
#endif

        timer->startTimer(3, true);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            cellIndex,
            particleIndex,
            (float4 *) pos,
            (float4 *) tempPos,
            (float4 *) posAfterLastSort,
            (float4 *) vel,
            (float4 *) tempVel,
            pointHasMovedMoreThanThreshold,
            numParticles);
        timer->stopTimer(3, true);
        *posAfterLastSortIsValid = true;
#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(tempPosTex));
        checkCudaErrors(cudaUnbindTexture(tempVelTex));
#endif

        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    }

    void collide(float *pos,
                 float *vel,
                 float *force,
                 uint  *cellIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint  *numNeighbors,
                 uint   numParticles,
                 uint   numCells,
                 EventTimer* timer)
    {
        if (numParticles == 0) {
          return;
        }
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        timer->startTimer(4, true);
        collideD<<< numBlocks, numThreads >>>((float4 *)pos,
                                              (float4 *)vel,
                                              (float4 *)force,
                                              cellIndex,
                                              cellStart,
                                              cellEnd,
                                              numParticles,
                                              numNeighbors);
        timer->stopTimer(4, true);
    
        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(posTex));
        checkCudaErrors(cudaUnbindTexture(velTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
    }

    bool checkForResort(bool *pointHasMovedMoreThanThreshold)
    {
      bool needsResort;
      cudaMemcpy(&needsResort, pointHasMovedMoreThanThreshold, sizeof(bool), cudaMemcpyDeviceToHost);
      return needsResort;
    }

}   // extern "C"
