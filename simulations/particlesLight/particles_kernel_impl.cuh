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

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> posTex;
texture<float4, 1, cudaReadModeElementType> velTex;

texture<float4, 1, cudaReadModeElementType> tempPosTex;
texture<float4, 1, cudaReadModeElementType> tempVelTex;

texture<float, 1, cudaReadModeElementType> nPosTex;
texture<float, 1, cudaReadModeElementType> nVelTex;

texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;


struct integrate_functor
{
    float _deltaTime;
    bool  _posAfterLastSortIsValid;
    bool *_pointHasMovedMoreThanThreshold;

    __host__ __device__
    integrate_functor(float deltaTime, bool posAfterLastSortIsValid, bool *pointHasMovedMoreThanThreshold)
        : _deltaTime(deltaTime),
          _posAfterLastSortIsValid(posAfterLastSortIsValid),
          _pointHasMovedMoreThanThreshold(pointHasMovedMoreThanThreshold) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 forceData = thrust::get<2>(t);
        volatile float4 posAfterLastSortData = thrust::get<3>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);
        float3 force = make_float3(forceData.x, forceData.y, forceData.z);
        float3 posAfterLastSort = make_float3(posAfterLastSortData.x, posAfterLastSortData.y, posAfterLastSortData.z);

        // How they initially calculated the new velocity
        vel += force * _deltaTime / 2;

        vel += params.gravity * _deltaTime;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * _deltaTime + 0.5 * force * _deltaTime * _deltaTime;

        vel += force * _deltaTime / 2;

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 4.0f - params.particleRadius)
        {
            pos.x = 4.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -4.0f + params.particleRadius)
        {
            pos.x = -4.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 4.0f - params.particleRadius)
        {
            pos.y = 4.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 4.0f - params.particleRadius)
        {
            pos.z = 4.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -4.0f + params.particleRadius)
        {
            pos.z = -4.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        if (pos.y < -4.0f + params.particleRadius)
        {
            pos.y = -4.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);

        if (_posAfterLastSortIsValid) {
            float3 movementSinceLastSort = pos - posAfterLastSort;
            float movementMagnitude = length(movementSinceLastSort);
            if (movementMagnitude >= params.movementThreshold) {
                *_pointHasMovedMoreThanThreshold = true;
            }
        }
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / (params.cellSize.x + 2*params.movementThreshold));
    gridPos.y = floor((p.y - params.worldOrigin.y) / (params.cellSize.y + 2*params.movementThreshold));
    gridPos.z = floor((p.z - params.worldOrigin.z) / (params.cellSize.z + 2*params.movementThreshold));
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcCellIndex(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;

}

// calculate cell index for each particle
__global__
void calcCellIndicesD(uint   *cellIndex,  // output
                      uint   *particleIndex, // output
                      float4 *pos,               // input: positions
                      uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcCellIndex(gridPos);

    // store grid hash and particle index
    cellIndex[index] = hash;
    particleIndex[index] = index;
}

// Copy pos and vel into temporary arrays
__global__
void copyArraysD(float4 *pos,
            float4 *tempPos,
            float4 *vel,
            float4 *tempVel,
            uint numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    float4 oldPos = FETCH(pos, index);
    float4 oldVel = FETCH(vel, index);

    tempPos[index] = oldPos;
    tempVel[index] = oldVel;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  uint   *cellIndex,        // input: sorted cell indices
                                  uint   *particleIndex,    // input: sorted particle indices
                                  float4 *pos,              // input: sorted position array
                                  float4 *tempPos,
                                  float4 *oldPos,           // Place to save old positions
                                  float4 *vel,              // input: sorted velocity array
                                  float4 *tempVel,
                                  bool   *pointHasMovedMoreThanThreshold, 
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    
    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = cellIndex[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = cellIndex[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // 0th thread resets the move threshold trigger for resort
        if (index == 0) {
            *pointHasMovedMoreThanThreshold = false;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = particleIndex[index];
        float4 threadPos = FETCH(tempPos, sortedIndex);
        float4 threadVel = FETCH(tempVel, sortedIndex);

        pos[index] = threadPos;
        oldPos[index] = threadPos;
        vel[index] = threadVel;
    }
}

// Find the start of each cell in the sorted cell indices
__global__
void findCellStartD(uint   *cellStart,        // output: cell start index
                    uint   *cellEnd,          // output: cell end index
                    uint   *cellIndex,        // input: sorted cell indices
                    float4 *pos,
                    float4 *oldPos,           // Place to save old positions
                    uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = cellIndex[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = cellIndex[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now, store the oldPositions. 
        float4 particlePos = FETCH(pos, index);   // macro does either global read or texture fetch

        oldPos[index] = particlePos;
    }
}

__global__
void reorderPreCollideD(float4 *pos,
                   float4 *vel,
                   float *tempPos,
                   float *tempVel,
                   uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index < numParticles) {
        tempPos[index] = pos[index].x;
        tempPos[index + numParticles] = pos[index].y;
        tempPos[index + 2 * numParticles] = pos[index].z;
        tempVel[index] = vel[index].x;
        tempVel[index + numParticles] = vel[index].y;
        tempVel[index + 2 * numParticles] = vel[index].z;
    }

}

__global__
void reorderPostCollideD(float4 *force,
                    float *tempForce,
                    uint numParticles)
{
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    if (index < numParticles) {
        force[index].x = tempForce[index];
        force[index].y = tempForce[index + numParticles];
        force[index].z = tempForce[index + 2 * numParticles];

    }

}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  particlePos,
                   float3  particleVel,
                   float  *nPos,
                   float  *nVel,
                   uint   *cellStart,
                   uint   *cellEnd,
                   uint    numParticles,
                   uint*   numNeighbors)
{
    uint cellIndex = calcCellIndex(gridPos);
    //uint neighbors = 0; 

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, cellIndex);

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, cellIndex);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {

                float3 pos2;
                pos2.x = FETCH(nPos, j);
                pos2.y = FETCH(nPos, j + numParticles);
                pos2.z = FETCH(nPos, j + 2 * numParticles);
            
                float3 vel2;
                vel2.x = FETCH(nVel, j);
                vel2.y = FETCH(nVel, j + numParticles);
                vel2.z = FETCH(nVel, j + 2 * numParticles);

                // collide two spheres
                force += collideSpheres(particlePos, pos2, particleVel, vel2, params.particleRadius, 
                                        params.particleRadius, params.attraction);
                //++neighbors; 
            }
        }
    }
    //numNeighbors[index + 1] = neighbors; 

    return force;
}

__global__
void collideD(float *nPos,               // input: position
              float *nVel,               // input: velocity
              float *force,             // output: forces
              uint   *cellIndex,    
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
              uint*   numNeighbors)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // Make this a command line flag at some point
    /*if (index == 0) {
        ++numNeighbors[0]; 
    }
    numNeighbors[index + 1] = 0; */

    // read particle data from sorted arrays
    float3 particlePos;
    particlePos.x = FETCH(nPos, index);
    particlePos.y = FETCH(nPos, index + numParticles);
    particlePos.z = FETCH(nPos, index + 2 * numParticles);

    float3 particleVel;
    particleVel.x = FETCH(nVel, index);
    particleVel.y = FETCH(nVel, index + numParticles);
    particleVel.z = FETCH(nVel, index + 2 * numParticles);

    // get address in grid
    int3 gridPos = calcGridPos(particlePos);

    // examine neighbouring cells
    float3 particleForce = make_float3(0.0f);

    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x, y, z);
                particleForce += collideCell(neighborPos, index, particlePos, particleVel, nPos, nVel, 
                                             cellStart, cellEnd, numParticles, numNeighbors);
            }
        }
    }

    // collide with cursor sphere
    particleForce += collideSpheres(particlePos, params.colliderPos, particleVel, 
                                    make_float3(0.0f, 0.0f, 0.0f), 
                                    params.particleRadius, params.colliderRadius, 0.0f);

    force[index] = particleForce.x;
    force[index + numParticles] = particleForce.y;
    force[index + 2 * numParticles] = particleForce.z;
}

#endif