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
        make_float3(objParams._cubeSize / 2.0 * objParams._voxelSize);

    // Divide by voxel size to get which voxel the particle is in
    // Voxel are numbered 0->cubeSize - 1, in the positive direction
    int3 voxelGridPos;
    voxelGridPos.x = floor((pos.x - lowerCorner.x) / objParams._voxelSize);
    voxelGridPos.y = floor((pos.y - lowerCorner.y) / objParams._voxelSize);
    voxelGridPos.z = floor((pos.z - lowerCorner.z) / objParams._voxelSize);

    return voxelGridPos;
}

__device__
uint getVoxelIndex(int3 voxelGridPos)
{
    return (voxelGridPos.z * objParams._cubeSize * objParams._cubeSize) + 
           (voxelGridPos.y * objParams._cubeSize) + voxelGridPos.x;
}

__device__
bool posIsOutOfBounds(int3 voxelGridPos)
{
    // Check if particle is in the bounding box of the object
    if (voxelGridPos.x < 0 || voxelGridPos.x >= objParams._cubeSize) {
        return true;
    }
    if (voxelGridPos.y < 0 || voxelGridPos.y >= objParams._cubeSize) {
        return true;
    }
    if (voxelGridPos.z < 0 || voxelGridPos.z >= objParams._cubeSize) {
        return true;
    }
    return false;
}

__device__
bool isActiveVoxel(int3 voxelGridPos, int* voxelStrength) 
{
    if (posIsOutOfBounds(voxelGridPos)) {
        return false;
    }
    else if (!voxelStrength[getVoxelIndex(voxelGridPos)]) {
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

__device__
float3 findNearestFaceNew(int3 voxelGridPos, float3 particlePos, float3 particleOldPos,
                          float3 voxelPos, int *voxelStrength) 
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
                                    return findNearestFaceNew(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
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
                                    return findNearestFaceNew(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
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
                                    return findNearestFaceNew(voxelGridPos, particlePos, newOldPos, voxelPos, voxelStrength);
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

__device__
float3 findNearestFace(int3 voxelGridPos, float3 particlePos, 
                       float3 particleVel, float3 voxelPos, 
                       int *voxelStrength) 
{
    float halfSide = objParams._voxelSize / 2.0;
    float penaltyDistance = objParams._voxelSize / 4.0; 
    float3 relativePos = particlePos - voxelPos; 

    // Calculate horizontal distance from each face
    // Penalize the distance if it's a face pointing to an active voxel
    // Or if the velocity indicates it probably didn't come from that direction.
    float distFromPosX = halfSide + voxelPos.x - particlePos.x; 
    if (isActiveVoxel(voxelGridPos + make_int3(1,0,0), voxelStrength)) {
        distFromPosX += penaltyDistance;
    }
    if (particleVel.x > 0) {
        distFromPosX += penaltyDistance;
    }
    float distFromNegX = objParams._voxelSize - distFromPosX; 
    if (isActiveVoxel(voxelGridPos + make_int3(-1,0,0), voxelStrength)) {
        distFromNegX += penaltyDistance;
    }
    if (particleVel.x < 0) {
        distFromNegX += penaltyDistance;
    }
    float distFromPosY = halfSide + voxelPos.y - particlePos.y;
    if (isActiveVoxel(voxelGridPos + make_int3(0,1,0), voxelStrength)) {
        distFromPosY += penaltyDistance;
    }
    if (particleVel.y > 0) {
        distFromPosY += penaltyDistance;
    }
    float distFromNegY = objParams._voxelSize - distFromPosY;
    if (isActiveVoxel(voxelGridPos + make_int3(0,-1,0), voxelStrength)) {
        distFromNegY += penaltyDistance;
    }
    if (particleVel.y < 0) {
        distFromNegY += penaltyDistance;
    }

    float distFromPosZ = halfSide + voxelPos.z - particlePos.z;
    if (isActiveVoxel(voxelGridPos + make_int3(0,0,1), voxelStrength)) {
        distFromPosZ += penaltyDistance;
    }
    if (particleVel.z > 0) {
        distFromPosZ += penaltyDistance;
    }
    float distFromNegZ = objParams._voxelSize - distFromPosZ;
    if (isActiveVoxel(voxelGridPos + make_int3(0,0,-1), voxelStrength)) {
        distFromNegZ += penaltyDistance;
    }
    if (particleVel.z < 0) {
        distFromNegZ += penaltyDistance;
    }

    float3 direction = make_float3(0.0f);

    // Find which x,y, and z face is closest
    // Set corresponding component of directional vector
    // to 1 or -1 based on which is closer
    float minXDist, minYDist, minZDist;
    if (distFromPosX < distFromNegX) {
        direction.x = 1.0;
        minXDist = distFromPosX;
    } else {
        direction.x = -1.0;
        minXDist = distFromNegX;
    }
    if (distFromPosY < distFromNegY) {
        direction.y = 1.0;
        minYDist = distFromPosY;
    } else {
        direction.y = -1.0;
        minYDist = distFromNegY;
    }
    if (distFromPosZ < distFromNegZ) {
        direction.z = 1.0;
        minZDist = distFromPosZ;
    } else {
        direction.z = -1.0;
        minZDist = distFromNegZ;
    }

    // Based on min x, y, z, find closest face, and return unit vector to it

    // X is shortest
    if (minXDist < minYDist && minXDist < minZDist) {
        direction = make_float3(direction.x, 0.0, 0.0);
    }
    // Check for if Y is shortest - only need to compare against Z because of previous X check
    else if (minYDist < minZDist) {
        direction = make_float3(0.0, direction.y, 0.0);
    }
    // Otherwise, Z must be shortest
    else {
        direction = make_float3(0.0, 0.0, direction.z);
    }
    return direction;
}

// Used for soft cube collision - not currently working as well as hard cube
__device__
float3 calcForceFromVoxel(int3 voxelGridPos, 
                          float4 *voxelPos,
                          int  *voxelStrength,  
                          float3 particlePos,
                          float3 particleVel,
                          float3 force)
{

    // Check that the particle is active/ in the object
    if (!isActiveVoxel(voxelGridPos, voxelStrength)) {
        return make_float3(0.0f);
    }

    // Find the direction the force should act in
    // Which is a unit vector pointing to the nearest face
    float3 voxelCenter = make_float3(voxelPos[getVoxelIndex(voxelGridPos)]);
    float3 direction = findNearestFace(voxelGridPos, particlePos, 
                                       particleVel, voxelCenter, voxelStrength);

    // Take component of velocity in same direction as nearest face
    particleVel.x = particleVel.x * (1 && direction.x);
    particleVel.y = particleVel.y * (1 && direction.y);
    particleVel.z = particleVel.z * (1 && direction.z);

    // Magnitude is based on distance from the center of the voxel
    float magnitude = objParams._voxelSize - 
                      dot(particlePos - voxelCenter, direction);

    // Force is magic number plus adjusted velocity times the mag times the dir
    float3 forceFromVoxel = (10 + 10 * length(particleVel)) * 
                             magnitude * direction; 

    return forceFromVoxel;
}



// Particle System Kernels/Functions

__global__
void integrateSystemD(float4 *pos,
                 float4 *vel,
                 float4 *force,
                 float4 *posAfterLastSort, 
                 float deltaTime,
                 uint numParticles, 
                 float4 * voxelPos, 
                 int *voxelStrength,  
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
    const float3 threadOldPos = make_float3(posAfterLastSort[index]);

    // Do first half of verlet integration
    threadVel += threadForce * deltaTime / 2;
    threadVel += params.gravity * deltaTime;
    threadVel *= params.globalDamping;
    
    // new position = old position + velocity * deltaTime
    // Second part of verlet integration
    threadPos += threadVel * deltaTime + 0.5 * threadForce * deltaTime * deltaTime;
    threadVel += (threadForce * deltaTime / 2);

#if USE_HARD_CUBE
    if (params.usingObject) {
        // Check that containing voxel is active
        const int3 voxelGridPos = getVoxelGridPos(threadPos);
        if (isActiveVoxel(voxelGridPos, voxelStrength)) {
            // Do the collision
            uint voxelIndex = getVoxelIndex(voxelGridPos);
            const float3 currentVoxelPos = make_float3(voxelPos[voxelIndex]);
            const float3 direction = findNearestFaceNew(voxelGridPos, threadPos, threadOldPos, currentVoxelPos, voxelStrength); 
            //if (index %100 == 0) printf("Vel: %f, %f, %f Dir:%f, %f, %f\n", threadVel.x, threadVel.y, threadVel.z, direction.x, direction.y, direction.z);
            //printf("%f, %f, %f\n", direction.x, direction.y, direction.z);
            // Update strength of voxel post-collision
#if 1
            int strength = voxelStrength[voxelIndex];
            if (strength > 0) {
                // Faster particles do more damage
                int newStrength = max(strength - (int) (10 * (length(threadVel)/0.5)), 0);
                atomicMin(&voxelStrength[voxelIndex], newStrength);
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
    if (threadPos.y > 4.0f - params.particleRadius)
    {
        threadPos.y = 4.0f - params.particleRadius;
        threadVel.y *= params.boundaryDamping;
    }
    if (threadPos.z > 4.0f - params.particleRadius)
    {
        threadPos.z = 4.0f - params.particleRadius;
        threadVel.z *= params.boundaryDamping;
    }
    if (threadPos.z < -4.0f + params.particleRadius)
    {
        threadPos.z = -4.0f + params.particleRadius;
        threadVel.z *= params.boundaryDamping;
    }

#endif

    if (threadPos.y < -4.0f + params.particleRadius)
    {
        threadPos.y = -4.0f + params.particleRadius;
        threadVel.y *= params.boundaryDamping;
    }

    // Check for particles to be removed by height or time - filter out those 
    // That were already removed by collision with cube
    if ((iters < 2 * params.maxIterations) && 
        ((params.limitParticleLifeByTime && (iters >= params.maxIterations)) ||
        (params.limitParticleLifeByHeight && (threadPos.x < params.maxDistance))))
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
                   float4 *pos,
                   float4 *vel,
                   uint   *cellStart,
                   uint   *cellEnd,
                   uint*   numNeighbors)
{
    uint cellIndex = calcCellIndex(gridPos);
    uint neighbors = 0; 

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
                float3 pos2 = make_float3(FETCH(pos, j));
                float3 vel2 = make_float3(FETCH(vel, j));

                // collide two spheres
                force += collideSpheres(particlePos, pos2, particleVel, vel2, params.particleRadius, 
                                        params.particleRadius, params.attraction);
                ++neighbors; 
            }
        }
    }
    numNeighbors[index + 1] = neighbors; 

    return force;
}

__global__
void collideD(float4 *pos,               // input: position
              float4 *vel,               // input: velocity
              float4 *force,             // output: forces
              int   *voxelStrength,
              float4 *voxelPos,
              uint   *cellIndex,    
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
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
                float3 tempForce = collideCell(neighborPos, index, particlePos, particleVel, pos, vel, 
                                             cellStart, cellEnd, numNeighbors);
                particleForce += tempForce;
            }
        }
    }

#if !USE_HARD_CUBE
    if (params.usingObject) {
        // Check for collisions with voxel object
        int3 voxelGridPos = getVoxelGridPos(particlePos);

        float3 forceFromObject = calcForceFromVoxel(voxelGridPos, voxelPos, voxelStrength, particlePos, particleVel, particleForce);
        particleForce += forceFromObject;
    }
#endif

    // collide with cursor sphere
    particleForce += collideSpheres(particlePos, params.colliderPos, particleVel, 
                                    make_float3(0.0f, 0.0f, 0.0f), 
                                    params.particleRadius, params.colliderRadius, 0.0f);

    // Note: Is 1.0 reasonable???
    force[index] = make_float4(particleForce, 1.0f);
}

#endif
