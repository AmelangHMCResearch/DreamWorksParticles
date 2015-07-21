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

texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;

texture<uint, 1, cudaReadModeElementType> voxelStrengthTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;
#endif

// simulation parameters in constant memory
__constant__ SimParams params;
__constant__ ObjectParams objParams;

// VoxelObject Functions:
__device__
int3 getVoxelGridPos(const float3 pos)
{
    // Find pos of particle with respect to origin of object
    const float3 lowerCorner = objParams._origin -
        (make_float3(objParams._cubeSize) / (float) 2.0 * objParams._voxelSize);

    // Divide by voxel size to get which voxel the particle is in
    // Voxel are numbered 0->cubeSize - 1, in the positive direction
    int3 voxelGridPos;
    voxelGridPos.x = floor((pos.x - lowerCorner.x) / objParams._voxelSize);
    voxelGridPos.y = floor((pos.y - lowerCorner.y) / objParams._voxelSize);
    voxelGridPos.z = floor((pos.z - lowerCorner.z) / objParams._voxelSize);

    return voxelGridPos;
}

__device__
uint3 calculateVoxelGridPosFromIndex(uint index)
{
    uint3 center;
    center.z = index / (objParams._cubeSize.x * objParams._cubeSize.y);
    center.y = (index - center.z * objParams._cubeSize.x * objParams._cubeSize.y) / objParams._cubeSize.x; 
    center.x = index - objParams._cubeSize.x * (center.y + objParams._cubeSize.y * center.z);
    return center;
}

__device__
uint3 calculateCubePosFromIndex(uint index)
{
    uint3 center;
    center.z = index / ((objParams._cubeSize.x +1) * (objParams._cubeSize.y + 1));
    center.y = (index - center.z * (objParams._cubeSize.x + 1) * (objParams._cubeSize.y + 1)) / (objParams._cubeSize.x + 1); 
    center.x = index - (objParams._cubeSize.x + 1) * (center.y + (objParams._cubeSize.y + 1) * center.z);
    return center;
}

__device__
float3 calculateVoxelCenter(int3 gridPos)
{
    float3 center;
    center.x = objParams._origin.x + (objParams._voxelSize / 2.0) + (gridPos.x - objParams._cubeSize.x / 2.0) * objParams._voxelSize;
    center.y = objParams._origin.y + (objParams._voxelSize / 2.0) + (gridPos.y - objParams._cubeSize.y / 2.0) * objParams. _voxelSize;
    center.z = objParams._origin.z + (objParams._voxelSize / 2.0) + (gridPos.z - objParams._cubeSize.z / 2.0) * objParams._voxelSize;
    return center;
}


__device__
uint getVoxelIndex(int3 voxelGridPos)
{
    return (voxelGridPos.z * objParams._cubeSize.x * objParams._cubeSize.y) + 
           (voxelGridPos.y * objParams._cubeSize.x) + voxelGridPos.x;
}

__device__
bool posIsOutOfBounds(int3 voxelGridPos)
{
    // Check if particle is in the bounding box of the object
    if (voxelGridPos.x < 0 || voxelGridPos.x >= objParams._cubeSize.x) {
        return true;
    }
    if (voxelGridPos.y < 0 || voxelGridPos.y >= objParams._cubeSize.y) {
        return true;
    }
    if (voxelGridPos.z < 0 || voxelGridPos.z >= objParams._cubeSize.z) {
        return true;
    }
    return false;
}

__device__
bool isActiveVoxel(int3 voxelGridPos, float* voxelStrength) 
{
    if (posIsOutOfBounds(voxelGridPos)) {
        return false;
    }
    else if (voxelStrength[getVoxelIndex(voxelGridPos)] <= 0.0f) {
        return false;
    }
    return true;
}

__device__
bool pointsIntersectFace(float knownCoord, float3 oldPos, float3 newPos) {
    // Must pass arguments such that oldPos.x and newPos.x are the same
    // Component as knownCoord. X, Y, and Z here don't correspond to the
    // real coordinates
    float y, z; 
    float t = (knownCoord - oldPos.x) / (newPos.x - oldPos.x); 
    y = oldPos.y + (newPos.y - oldPos.y) * t; 
    z = oldPos.z + (newPos.z - oldPos.z) * t;
    if (abs(y) < objParams._voxelSize && abs(z) < objParams._voxelSize) {
        return true;
    } else {
        return false;
    }


}


// This function is officially defunct given the erosion sim :()
__device__
float3 findNearestFace(int3 voxelGridPos, float3 particlePos, float3 particleOldPos,
                          float3 voxelPos, float *voxelStrength) 
{
    // Note: The current check on active voxels will break down when we have more complex geography
    // Consider ways to fix it. Also this is a nightmare for warp divergence :(
    float3 relOldPos = particleOldPos - voxelPos;
    float3 relNewPos = particlePos - voxelPos; 
    // For each of 6 faces: 
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++ y) {
            for (int x = -1; x <= 1; ++x) {
                if ((x==0 && y==0) || (x==0 && z==0) || (y==0 && z==0)) {
                    float3 direction = make_float3(x, y, z); 
                    // If we're checking an x face
                    if (direction.x != 0) {
                        // If the old face is on the outside of this face
                        if (direction.x * relOldPos.x > objParams._voxelSize / 2.0) {
                            // And if the line between the old and the new intersects the face
                            if (pointsIntersectFace(direction.x * objParams._voxelSize / 2.0, 
                                                     make_float3(relOldPos.x, relOldPos.y, relOldPos.z), 
                                                     make_float3(relNewPos.x, relNewPos.y, relNewPos.z))) {
                                // If this face is up against an active voxel, try again without that component of the vel. 
                                if (isActiveVoxel(voxelGridPos + make_int3(direction), voxelStrength)) {
                                    float3 newOldPos = make_float3(particlePos.x, particleOldPos.y, particleOldPos.z);
                                    return findNearestFace(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
                                }
                                return direction;
                            }
                        }
                    }
                    else if (direction.y != 0) {
                        if (direction.y * relOldPos.y > objParams._voxelSize / 2.0) {
                            if (pointsIntersectFace(direction.y * objParams._voxelSize / 2.0, 
                                                     make_float3(relOldPos.y, relOldPos.x, relOldPos.z), 
                                                     make_float3(relNewPos.y, relNewPos.x, relNewPos.z))) {
                                if (isActiveVoxel(voxelGridPos + make_int3(direction), voxelStrength)) {
                                    float3 newOldPos = make_float3(particleOldPos.x, particlePos.y, particleOldPos.z);
                                    return findNearestFace(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
                                }
                                return direction;
                            }
                        }
                    }
                    else if (direction.z != 0) {
                        if (direction.z * relOldPos.z > objParams._voxelSize / 2.0) {
                            if (pointsIntersectFace(direction.z * objParams._voxelSize / 2.0, 
                                                     make_float3(relOldPos.z, relOldPos.x, relOldPos.y), 
                                                     make_float3(relNewPos.z, relNewPos.x, relNewPos.y))) {
                                if (isActiveVoxel(voxelGridPos + make_int3(direction), voxelStrength)) {
                                    float3 newOldPos = make_float3(particleOldPos.x, particleOldPos.y, particlePos.z);
                                    return findNearestFace(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
                                }
                                return direction;
                            }
                        }
                    }
                }
            }
        }
    }
    return make_float3(0,0,0);
}

// Used for soft cube collision - not currently working as well as hard cube
__device__
float3 calcForceFromVoxel(float3 particlePos,
                          float3 voxelPos, 
                          float3 particleVel,
                          uint voxelIndex,
                          float deltaTime, 
                          float* voxelStrength)
{
    // calculate relative position
    float3 relPos = particlePos - voxelPos;

    //float relativeMass = voxelMass / particleMass;

    float dist = length(relPos);
    float collideDist = params.particleRadius;

    float3 forceOnParticle = make_float3(0.0f);

    if (dist < collideDist)
    {
        // Faster particles do more damage
        float t_c = 0.1; 
        float amountToReduceStrength = min(dot(particleVel, relPos), 0.0f) * deltaTime / t_c;
        atomicAdd(&voxelStrength[voxelIndex], amountToReduceStrength);

        forceOnParticle = (128 * objParams._voxelSize) * params.damping * (relPos) / dist * (params.particleRadius - dist) / params.particleRadius;
    }

    return forceOnParticle;
}



// Particle System Kernels/Functions
__global__
void integrateSystemD(float4 *pos,
                 float4 *vel,
                 float4 *force,
                 float4 *posAfterLastSort, 
                 float deltaTime,
                 uint numParticles, 
                 float *voxelStrength,  
                 bool posAfterLastSortIsValid, 
                 bool *pointHasMovedMoreThanThreshold,
                 uint *numParticlesToRemove)
{
    const uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // Grab the numbers that we need to from memory
    float3 threadPos          = make_float3(pos[index]);
    float3 threadVel          = make_float3(vel[index]);
    float iters = vel[index].w;
    float3 threadForce        = make_float3(force[index]);
    float density             = force[index].w;
    if (density == 0) {
        density = 1;
        if (iters != 0) {
            iters = params.maxIterations * 2; 
            atomicAdd(numParticlesToRemove, 1);
            *pointHasMovedMoreThanThreshold = true;
        }
    }
    const float3 threadOldPos = make_float3(posAfterLastSort[index]);

    // Do first half of verlet integration
    threadVel += threadForce / density * deltaTime / 2;
    threadVel += params.gravity * deltaTime;
    threadVel *= params.globalDamping;

    // new position = old position + velocity * deltaTime
    // Second part of verlet integration
    threadPos += threadVel * deltaTime + 0.5 * threadForce / density * deltaTime * deltaTime;
    threadVel += (threadForce / density * deltaTime / 2);

#if USE_HARD_CUBE
    if (params.usingObject) {
        // Check that containing voxel is active
        const int3 voxelGridPos = getVoxelGridPos(threadPos);
        if (isActiveVoxel(voxelGridPos, voxelStrength)) {
            // Do the collision
            uint voxelIndex = getVoxelIndex(voxelGridPos);
            float3 currentVoxelPos = calculateVoxelCenter(voxelGridPos);
            const float3 direction = findNearestFace(voxelGridPos, threadPos, threadOldPos, currentVoxelPos, voxelStrength); 

            // Update strength of voxel post-collision
#if 1
            float strength = voxelStrength[voxelIndex];
            if (strength > 0) {
                // Faster particles do more damage
                float amountToReduceStrength = abs(5.0 * (direction.x * threadVel.x * threadVel.x + direction.y * threadVel.y * threadVel.y + direction.z * threadVel.z * threadVel.z));
                //float amountToReduceStrength = 5.0 * threadVel.x * threadVel.x;
                atomicAdd(&voxelStrength[voxelIndex], -1.0 * amountToReduceStrength);
                /*float newStrength = max((float)(strength - (10 * (length(threadVel)/0.5))), 0.0f);
                atomicMin(&voxelStrength[voxelIndex], newStrength);*/
            }
#endif
            
            // If it's trying to move into an active particle, remove it
            int3 neighborVoxelGridPos = voxelGridPos + make_int3(direction);
            if (isActiveVoxel(neighborVoxelGridPos, voxelStrength)) {
                iters = params.maxIterations * 2; 
                atomicAdd(numParticlesToRemove, 1);
                *pointHasMovedMoreThanThreshold = true;
            }
            // Calculate new position and velocity
            float distanceToMove = objParams._voxelSize / 2.0 + 0.0001;
            if (direction.x != 0.0) {
                if (threadVel.x * direction.x < 0) {
                    threadVel.x *= -1.0;
                }
                threadPos.x = currentVoxelPos.x + direction.x * distanceToMove;
            }
            else if (direction.y != 0.0) {
                if (threadVel.y * direction.y < 0) {
                    threadVel.y *= -1.0;
                }
                threadPos.y = currentVoxelPos.y + direction.y * distanceToMove;
            }
            else {
                if (threadVel.z * direction.z < 0) {
                    threadVel.z *= -1.0;
                }
                threadPos.z = currentVoxelPos.z + direction.z * distanceToMove;
            }
              
        }
    }
#endif
    
    // set this to zero to disable collisions with cube top
#if 0
    if (threadPos.y > 0.5.0f - params.particleRadius)
    {
        threadPos.y = 0.5f - params.particleRadius;
        threadVel.y *= params.boundaryDamping;
    }
#endif

// set this to zero to disable collisions with cube sides
#if 1
    if (threadPos.x > 4.0f - params.particleRadius)
    {
        threadPos.x = 4.0f - params.particleRadius;
        threadVel.x *= params.boundaryDamping;
    }
    if (threadPos.x < -4.0f + params.particleRadius)
    {
        threadPos.x = -4.0f + params.particleRadius;
        threadVel.x *= params.boundaryDamping;
    }
    if (threadPos.z > 0.5f - params.particleRadius)
    {
        threadPos.z = 0.5f - params.particleRadius;
        threadVel.z *= params.boundaryDamping;
    }
    if (threadPos.z < -0.5f + params.particleRadius)
    {
        threadPos.z = -0.5f + params.particleRadius;
        threadVel.z *= params.boundaryDamping;
    }
#endif

    if (threadPos.y < -0.5f + params.particleRadius)
    {
        threadPos.y = -0.5f + params.particleRadius;
        threadVel.y *= params.boundaryDamping;
    }

    // Check for particles to be removed by height or time - filter out those 
    // That were already removed by collision with cube
    if ((iters < 2 * params.maxIterations) && 
        ((params.limitParticleLifeByTime && (iters >= params.maxIterations)) ||
        (params.limitParticleLifeByHeight && (threadPos.x > params.maxDistance))))
    {
        iters = params.maxIterations + 1; 
        atomicAdd(numParticlesToRemove, 1);
        *pointHasMovedMoreThanThreshold = true;
    }
    // Increment time particle has been alive if limiting by time
    if (params.limitParticleLifeByTime) {
        ++iters;
    }

    // store new position and velocity
    pos[index] = make_float4(threadPos, 1.0f);
    vel[index] = make_float4(threadVel, iters);
    if (posAfterLastSortIsValid) {
        float3 movementSinceLastSort = threadPos - threadOldPos;
        float movementMagnitude = length(movementSinceLastSort);
        if (movementMagnitude >= params.movementThreshold) {
            *pointHasMovedMoreThanThreshold = true;
        }
    }
}

// calculate position in uniform grid
__device__ 
int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / (params.cellSize.x + 2*params.movementThreshold));
    gridPos.y = floor((p.y - params.worldOrigin.y) / (params.cellSize.y + 2*params.movementThreshold));
    gridPos.z = floor((p.z - params.worldOrigin.z) / (params.cellSize.z + 2*params.movementThreshold));
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ 
uint calcCellIndex(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) +
           __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate address in grid from position (clamping to edges)
__device__ 
uint calcPreRemovalCellIndex(int3 gridPos, float numIters, float height)
{
    if (numIters > params.maxIterations) {
        return params.numCells;
    }
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + 
           __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate cell index for each particle
__global__
void calcCellIndicesD(uint   *cellIndex,  // output
                      uint   *particleIndex, // output
                      float4 *pos,               // input: positions
                      float4 *vel,
                      uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float3 p = make_float3(pos[index]);

    // get address in grid
    int3 gridPos = calcGridPos(p);
    uint hash = calcPreRemovalCellIndex(gridPos, vel[index].w, p.x);


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

// ____________Collision Functions_____________
// A couple smoothing kernels and their derivatives, used for the fluid simulation
__device__
float W(float r) {
    // Note: r should always be > 0
    float d = 16/64.0; 
    if (r > d) {
        return 0;
    } 
    float secondTerm = d * d - r * r; 
    return 0.01 * (315 / (64 * M_PI * (d * d * d * d * d * d * d * d * d))) * secondTerm * secondTerm * secondTerm; 
}

__device__
float gradW(float r) {
    float d = 16 / 64.0; 
    if (r > d) {
        return 0;
    }
    float secondTerm = d * d - r * r;
    return 0.003 * (315 / (64 * M_PI * (d * d * d * d * d * d * d * d * d))) * -6 * r * secondTerm * secondTerm; 
}

__device__
float laplacianW(float r) {
    float d = 1/1.0; 
    if (r > d) {
        return 0;
    }
    float secondTerm = d * d - r * r;
    return (6 * 315 / (64 * M_PI * (d * d * d * d * d * d * d * d * d))) * (-6 * secondTerm * secondTerm  + -4 * secondTerm);
}

__device__
float C(float r)
{
    float h = 16.0 / 64.0;
    if ((2 * r > h) && (r <= h)) {
        return 0.002 * 32.0 / (M_PI * h * h * h * h * h * h * h * h * h) * (h - r) * (h - r) * (h - r) * r * r * r;
    }
    else if ((r > 0) && (2 * r <= h)) {
        return 0.002 * 32.0 / (M_PI * h * h * h * h * h * h * h * h * h) * (h - r) * (h - r) * (h - r) * r * r * r - (h * h * h * h * h * h) / 64.0;
    } else {
        return 0;
    }
}

// This function is currently not used, but saving it just in case
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = 2 * radiusA + radiusB;

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
        force += 0.0005*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}



// Calculate the contribution of one cell to a particle's density
__device__
float addOneCellToDensity(int3    gridPos,
                           uint    index,
                           float3  particlePos,
                           float4 *pos,
                           uint   *cellStart,
                           uint   *cellEnd)
{
    uint cellIndex = calcCellIndex(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, cellIndex);

    float density = 0;
    float mass = 1;

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, cellIndex);

        for (uint j=startIndex; j<endIndex; j++)
        {
            float3 pos2 = make_float3(FETCH(pos, j));
            if (length(particlePos - pos2) < 10 * params.cellSize.x) {
                density += mass * W(length(particlePos - pos2)); 
            }
        }
    }
    return density;
}

// Calculate densities for all particles
__global__
void calcDensitiesD(float4 *pos,
                    float4 *force,
                    uint   *cellIndex,
                    uint   *cellStart,
                    uint   *cellEnd,
                    uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 particlePos = make_float3(FETCH(pos, index));

    // get address in grid
    int3 gridPos = calcGridPos(particlePos);

    float density = 0; 

    // Loop over other particles
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x, y, z);
                density += addOneCellToDensity(neighborPos, index, particlePos, pos, 
                                               cellStart, cellEnd);
            }
        }
    }

    // Note: Is 1.0 reasonable???
    force[index].w = density;
}

// Calculate from one cell's particles
__device__
float addOneCellToNormal(int3    gridPos,
                         uint    index,
                         float3  particlePos,
                         float4 *pos,
                         float  particleDensity,
                         float4 *force,
                         uint   *cellStart,
                         uint   *cellEnd)
{
    uint cellIndex = calcCellIndex(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, cellIndex);

    float normal = 0;
    float mass = 1;
    float h = 1;

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, cellIndex);

        for (uint j=startIndex; j<endIndex; j++)
        {
            float3 pos2 = make_float3(FETCH(pos, j));
            float density2 = force[j].w;

            if (length(particlePos - pos2) < 10 * params.cellSize.x) {
                normal += h * (mass / density2) * gradW(length(particlePos - pos2));
            }
        }
    }
    return normal;
}

__global__
void calcNormalsD(float4 *pos,
                  float4 *force,
                  float  *normals,
                  uint   *cellIndex,
                  uint   *cellStart,
                  uint   *cellEnd,
                  uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 particlePos = make_float3(FETCH(pos, index));
    float particleDensity = force[index].w;

    // get address in grid
    int3 gridPos = calcGridPos(particlePos);

    // examine neighbouring cells
    float normal = 0; 

    // Loop over neighboring cells
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x, y, z);
                normal += addOneCellToNormal(neighborPos, index, particlePos, pos, particleDensity, force,
                                               cellStart, cellEnd);
            }
        }
    }

    // Note: Is 1.0 reasonable???
    normals[index] = normal;


}

__device__
float3 calcViscousForce(float3 pos1, float3 pos2, float3 vel1, float3 vel2, float dens1, float dens2)
{
    float mu = 0.001;
    float mass = 1;
    return mu * mass * (vel2 - vel1) / dens2 * laplacianW(length(pos1 - pos2));
}

// This function is not currently in use, saved here in case needed in the future
__device__
float3 calcPressureForce(float3 pos1, float3 pos2, float dens1, float dens2)
{
    // First method: Based on muller tutorial
    float k = 0.0005;
    float ro_naught = 20.5;
    float mass = 1;
    float pressure1 = k * (dens1 - ro_naught);
    float pressure2 = k * (dens2 - ro_naught);
    float radius = length(pos1 - pos2);
    float3 dir = (pos1 - pos2) / radius;
    return -1.0 * dir * (mass * (pressure1 + pressure2) / (2 * dens2) * gradW(radius));

    // Second method based on Savannah making things up
    /*float radius = length(pos1 - pos2);
    float3 dir = (pos1 - pos2) / radius;
    float factor=0; 
    if (radius > 2 * params.particleRadius && radius < 4 * params.particleRadius) {
        factor = 200 * (radius - 2 * params.particleRadius) * (radius - 4 * params.particleRadius);
    }
    if (radius > 0 && radius <= 2 * params.particleRadius) {
        factor = 1.0 / radius - 1.0 / (2 * params.particleRadius);
    }
    return dir * factor * 0.01;*/
}

__device__
float3 calcSurfaceTensionForce(float3 pos1,
                               float3 pos2, 
                               float  norm1, 
                               float  norm2, 
                               float  density1, 
                               float  density2)
{
    // Constants for equations, which were chosen
    // based on what looked good
    float mass1 = 1;
    float mass2 = 1;
    float gamma = 0.001;
    float ro_naught = 20.5;

    // See Akinci et al, 2013 for explanation of equations
    float3 dir = (pos1 - pos2) / length(pos1 - pos2);
    float3 force1 = -1.0 * gamma * mass1 * mass2 * C(length(pos1 - pos2)) * dir;
    float3 force2 = -1.0 * gamma * mass1 * (norm1 - norm2) * dir;
    float k = (2 * ro_naught) / (density1 + density2);
    return k * (force1 + force2);
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  particlePos,
                   float3  particleVel,
                   float4 *pos,
                   float4 *vel,
                   float4 *force,
                   float  *normals,
                   uint   *cellStart,
                   uint   *cellEnd,
                   uint*   numNeighbors)
{
    uint cellIndex = calcCellIndex(gridPos);
    uint neighbors = 0; 
    float particleDensity = force[index].w;
    float particleNormal = normals[index];

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, cellIndex);

    float3 particleForce = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, cellIndex);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index) {
                float3 pos2 = make_float3(FETCH(pos, j));
                float3 vel2 = make_float3(FETCH(vel, j));
                float normal2 = normals[j];
                float density2 = force[j].w;
                particleForce += collideSpheres(particlePos, pos2, particleVel, vel2, params.particleRadius, params.particleRadius, params.attraction);

                // Check that we aren't looking at a far away particle, then find viscous + tension forces
                if (length(particlePos - pos2) < 10 * params.cellSize.x) {
                    particleForce += calcViscousForce(particlePos, pos2, particleVel, vel2, particleDensity, density2);
                    //particleForce += calcSurfaceTensionForce(particlePos, pos2, particleNormal, normal2, particleDensity, density2);
                }

                ++neighbors; 
            }
        }
    }
    numNeighbors[index + 1] = neighbors; 

    return particleForce;
}

__global__
void collideD(float4 *pos,               // input: position
              float4 *vel,               // input: velocity
              float4 *force,             // output: forces
              float   *voxelStrength,
              float   *normals,
              uint    *cellIndex,    
              uint    *cellStart,
              uint    *cellEnd,
              uint    numParticles,
              float   deltaTime,
              uint*   numNeighbors)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // Make this a command line flag at some point
    if (index == 0) {
        ++numNeighbors[0]; 
    }
    numNeighbors[index + 1] = 0; 

    // read particle data from sorted arrays
    float3 particlePos = make_float3(FETCH(pos, index));
    float3 particleVel = make_float3(FETCH(vel, index));
    float density = force[index].w;

    // get address in grid
    int3 gridPos = calcGridPos(particlePos);

    // examine neighbouring cells
    float3 particleForce = make_float3(0.0f);

    // Collide with other Particles
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighborPos = gridPos + make_int3(x, y, z);
                float3 tempForce = collideCell(neighborPos, index, particlePos, particleVel, pos, vel, force, normals,
                                             cellStart, cellEnd, numNeighbors);
                particleForce += tempForce;
            }
        }
    }

#if !USE_HARD_CUBE
    if (params.usingObject) {
        // Check for collisions with voxel object
        int3 voxelGridPos = getVoxelGridPos(particlePos);

        float3 forceFromObject = make_float3(0);
        uint numNeighboringVoxels = 0; 

        // Check all voxels which might intersecting the cube
        int loopStart = -floor(params.particleRadius / objParams._voxelSize);
        int loopEnd = ceil(params.particleRadius / objParams._voxelSize);
        for (int z = loopStart; z <= loopEnd; ++z) {
            for (int y = loopStart; y <= loopEnd; ++y) {
                for (int x = loopStart; x <= loopEnd; ++x) {
                    int3 neighborGridPos = voxelGridPos + make_int3(x, y, z);
                    if (isActiveVoxel(neighborGridPos, voxelStrength)) {
                        float3 voxelPosition = calculateVoxelCenter(neighborGridPos);
                        forceFromObject += calcForceFromVoxel(particlePos, voxelPosition, particleVel, getVoxelIndex(neighborGridPos), deltaTime, voxelStrength);
                        ++numNeighboringVoxels;
                    }
                }
            }
        }
        if (numNeighboringVoxels >= (loopEnd - loopStart) * (loopEnd - loopStart)) {
            density = 0; 
        }
        particleForce += forceFromObject;
    }
#endif

    // collide with cursor sphere
    particleForce += collideSpheres(particlePos, params.colliderPos, particleVel, 
                                    make_float3(0.0f, 0.0f, 0.0f), 
                                    params.particleRadius, params.colliderRadius, 0.0f);

    // Use 4th element of force to hold the density
    force[index] = make_float4(particleForce, density);
}

__device__
float tangle(float x, float y, float z)
{
    x *= 3.0f;
    y *= 3.0f;
    z *= 3.0f;
    return (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;
}

// evaluate field function at a point
// returns value and gradient in float4
__device__
float4 fieldFunc4(float3 p)
{
    float v = tangle(p.x, p.y, p.z);
    const float d = 0.001f;
    float dx = tangle(p.x + d, p.y, p.z) - v;
    float dy = tangle(p.x, p.y + d, p.z) - v;
    float dz = tangle(p.x, p.y, p.z + d) - v;
    return make_float4(dx, dy, dz, v);
}

__device__
void vertexInterp2(float isolevel, float3 p0, float3 p1, float4 f0, float4 f1, float3 &p, float3 &n)
{
    float t = (isolevel - f0.w) / (f1.w - f0.w);
    p = lerp(p0, p1, 0.5);
    n.x = lerp(f0.x, f1.x, t);
    n.y = lerp(f0.y, f1.y, t);
    n.z = lerp(f0.z, f1.z, t);
    n = normalize(n);
}

__global__
void createMarchingCubesMeshD(float4 *vertexPos,
                              float4 *norm,
                              float *voxelStrength,
                              uint  *tri,
                              uint  *numVerts,
                              uint  *numVerticesClaimed,
                              uint   numVoxelsToDraw)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    // Get gridPos of our grid cube - starts as lower left corner as (0,0,0)
    uint3 gridPos = calculateCubePosFromIndex(index);
    
    // Check if voxels on corner of gridcube are 
    int lookupIndexForActiveVertices = 0;
    float3 cubeVertexPos[8];
    float4 field[8];

    int3 i = make_int3(-1, -1, -1);
    int3 toCheck = make_int3(gridPos) + i;
    bool isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 0); 
    cubeVertexPos[0] = calculateVoxelCenter(toCheck);
    field[0] = fieldFunc4(cubeVertexPos[0]);

    i = make_int3(0,-1,-1);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 1); 
    cubeVertexPos[1] = calculateVoxelCenter(toCheck);
    field[1] = fieldFunc4(cubeVertexPos[1]);

    i = make_int3(0,0,-1);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 2); 
    cubeVertexPos[2] = calculateVoxelCenter(toCheck);
    field[2] = fieldFunc4(cubeVertexPos[2]);

    i = make_int3(-1,0,-1);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 3); 
    cubeVertexPos[3] = calculateVoxelCenter(toCheck);
    field[3] = fieldFunc4(cubeVertexPos[3]);

    i = make_int3(-1,-1,0);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 4); 
    cubeVertexPos[4] = calculateVoxelCenter(toCheck);
    field[4] = fieldFunc4(cubeVertexPos[4]);

    i = make_int3(0,-1,0);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 5); 
    cubeVertexPos[5] = calculateVoxelCenter(toCheck);
    field[5] = fieldFunc4(cubeVertexPos[5]);

    i = make_int3(0,0,0);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 6); 
    cubeVertexPos[6] = calculateVoxelCenter(toCheck);
    field[6] = fieldFunc4(cubeVertexPos[6]);

    i = make_int3(-1,0,0);
    toCheck = make_int3(gridPos) + i;
    isActive = isActiveVoxel(toCheck, voxelStrength);
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 7); 
    cubeVertexPos[7] = calculateVoxelCenter(toCheck);
    field[7] = fieldFunc4(cubeVertexPos[7]);




    /*for (int z = -1; z < 1; ++z) {
        for (int y = -1; y < 1; ++y) {
            for (int x = -1; x < 1; ++x) {
                int loopIndex = (x+1) + (y+1) * 2 + (z+1) * 4;
                int3 voxelToCheck = make_int3(gridPos) + make_int3(x, y, z); 
                bool voxelIsActive = isActiveVoxel(voxelToCheck, voxelStrength);
                lookupIndexForActiveVertices = lookupIndexForActiveVertices | 
                                               (voxelIsActive << loopIndex); 
                cubeVertexPos[loopIndex] = calculateVoxelCenter(voxelToCheck);
            }
        }
    }*/

    float3 vertlist[12];
    float3 normList[12];

    vertexInterp2(0.0, cubeVertexPos[0], cubeVertexPos[1], field[0], field[1], vertlist[0], normList[0]);
    vertexInterp2(0.0, cubeVertexPos[1], cubeVertexPos[2], field[1], field[2], vertlist[1], normList[1]);
    vertexInterp2(0.0, cubeVertexPos[2], cubeVertexPos[3], field[2], field[3], vertlist[2], normList[2]);
    vertexInterp2(0.0, cubeVertexPos[3], cubeVertexPos[0], field[3], field[0], vertlist[3], normList[3]);

    vertexInterp2(0.0, cubeVertexPos[4], cubeVertexPos[5], field[4], field[5], vertlist[4], normList[4]);
    vertexInterp2(0.0, cubeVertexPos[5], cubeVertexPos[6], field[5], field[6], vertlist[5], normList[5]);
    vertexInterp2(0.0, cubeVertexPos[6], cubeVertexPos[7], field[6], field[7], vertlist[6], normList[6]);
    vertexInterp2(0.0, cubeVertexPos[7], cubeVertexPos[4], field[7], field[4], vertlist[7], normList[7]);

    vertexInterp2(0.0, cubeVertexPos[0], cubeVertexPos[4], field[0], field[4], vertlist[8], normList[8]);
    vertexInterp2(0.0, cubeVertexPos[1], cubeVertexPos[5], field[1], field[5], vertlist[9], normList[9]);
    vertexInterp2(0.0, cubeVertexPos[2], cubeVertexPos[6], field[2], field[6], vertlist[10], normList[10]);
    vertexInterp2(0.0, cubeVertexPos[3], cubeVertexPos[7], field[3], field[7], vertlist[11], normList[11]);

    uint numVerticesToAdd = FETCH(numVerts, lookupIndexForActiveVertices);
    uint positionToAdd = atomicAdd(numVerticesClaimed, numVerticesToAdd); 
    for (int i= 0; i < numVerticesToAdd; ++i) {

        uint edge = FETCH(tri, lookupIndexForActiveVertices*16 + i);
        uint indexToAdd = positionToAdd + i;

        if (indexToAdd < numVoxelsToDraw * 15)
        {
            vertexPos[indexToAdd] = make_float4(vertlist[edge], 1.0f);
            norm[indexToAdd] = make_float4(normList[edge], 0.0f);
        }
    }

}
#endif
