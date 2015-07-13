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

    // Divide by voxel size to get which voxel in each direction the particle is in
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
    return (voxelGridPos.z * objParams._cubeSize * objParams._cubeSize) + (voxelGridPos.y * objParams._cubeSize) + voxelGridPos.x;
}

__device__
bool posIsOutOfBounds(int3 voxelGridPos)
{
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
float3 findNearestFace(int3 voxelGridPos, float3 particlePos, float3 voxelPos) {
    float halfSide = objParams._voxelSize / 2.0;
    float3 relativePos = particlePos - voxelPos; 

    // Calculate horizontal distance from each face
    float distFromPosX = halfSide + voxelPos.x - particlePos.x; 
    float distFromNegX = objParams._voxelSize - distFromPosX; 

    float distFromPosY = halfSide + voxelPos.y - particlePos.y;
    float distFromNegY = objParams._voxelSize - distFromPosY;

    float distFromPosZ = halfSide + voxelPos.z - particlePos.z;
    float distFromNegZ = objParams._voxelSize - distFromPosZ;

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
    //printf("X: %f Y: %f Z: %f\n", minXDist, minYDist, minZDist);

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

// Can't write to velocity during collide - fix soft cube collision later
__device__
float3 calcForceFromVoxel(int3 voxelGridPos, 
                          float4 *voxelPos,
                          bool  *activeVoxel,  
                          float3 particlePos,
                          float3 particleVel,
                          float3 force)
{

    // Check that the particle is in the bounding box of the object
    if (posIsOutOfBounds(voxelGridPos)) {
        return make_float3(0.0f);
    }
    // Check that the particle is in an active voxel
    uint voxelIndex = getVoxelIndex(voxelGridPos);
    if (!activeVoxel[voxelIndex]) {
        return make_float3(0.0f);
    }

    // Find the direction the force should act in
    // Which is a unit vector pointing to the nearest face
    float3 voxelCenter = make_float3(voxelPos[voxelIndex]);
    float3 direction = findNearestFace(voxelGridPos, particlePos, voxelCenter);

    // Take component of velocity in same direction as nearest face
    particleVel.x = particleVel.x * (1 && direction.x);
    particleVel.y = particleVel.y * (1 && direction.y);
    particleVel.z = particleVel.z * (1 && direction.z);

    // Magnitude is based on distance from the center of the voxel
    float magnitude = objParams._voxelSize - dot(particlePos - voxelCenter, direction);

    // Force is magic number plus adjusted velocity times the mag times the dir
    float3 forceFromVoxel = (10 + 10 * length(particleVel)) * magnitude * direction; 

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
                 bool *activeVoxel,  
                 bool posAfterLastSortIsValid, 
                 bool *pointHasMovedMoreThanThreshold,
                 uint *numParticlesToRemove)
{
    const uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    float3 threadPos          = make_float3(pos[index]);
    float3 threadVel          = make_float3(vel[index]);
    float iters = vel[index].w;
    float3 threadForce        = make_float3(force[index]);
    float density             = force[index].w;
    if (density == 0) {
        density = 1;
    }
    const float3 threadOldPos = make_float3(posAfterLastSort[index]);

    threadVel += threadForce / density * deltaTime / 2;
    threadVel += params.gravity * deltaTime;

    // new position = old position + velocity * deltaTime
    threadPos += threadVel * deltaTime + 0.5 * threadForce / density * deltaTime * deltaTime;
    threadVel += (threadForce / density * deltaTime / 2);

#if USE_HARD_CUBE
    if (params.usingObject) {
        // Check that the particle is in the bounding box of the object
        const int3 voxelGridPos = getVoxelGridPos(threadPos);
        if (!posIsOutOfBounds(voxelGridPos)) {
            // Check that the voxel the particle is in is active
            const uint voxelIndex = getVoxelIndex(voxelGridPos);
            if (activeVoxel[voxelIndex]) {
                // Do the collision
                const float3 currentVoxelPos = make_float3(voxelPos[voxelIndex]);
                const float3 direction = findNearestFace(voxelGridPos, threadPos, currentVoxelPos);

                int3 neighborVoxelGridPos = make_int3(voxelGridPos.x + (int) direction.x, voxelGridPos.y + (int) direction.y, voxelGridPos.z + (int) direction.z);
                if (!posIsOutOfBounds(neighborVoxelGridPos)) {
                    if (activeVoxel[getVoxelIndex(neighborVoxelGridPos)]) {
                        iters = params.maxIterations * 2; 
                        atomicAdd(numParticlesToRemove, 1);
                        *pointHasMovedMoreThanThreshold = true;
                    }
                }
                if (direction.x != 0.0) {
                    if (threadVel.x * direction.x < 0) {
                        threadVel.x *= -1.0 * params.globalDamping;
                    }
                    threadPos.x = currentVoxelPos.x + direction.x * (objParams._voxelSize / 2.0 + 0.0001);
                }
                else if (direction.y != 0.0) {
                    if (threadVel.y * direction.y < 0) {
                        threadVel.y *= -1.0 * params.globalDamping;
                    }
                    threadPos.y = currentVoxelPos.y + direction.y * (objParams._voxelSize / 2.0 + 0.0001);
                }
                else {
                    if (threadVel.z * direction.z < 0) {
                        threadVel.z *= -1.0 * params.globalDamping;
                    }
                    threadPos.z = currentVoxelPos.z + direction.z * (objParams._voxelSize / 2.0 + 0.0001);
                }
              
            } 
        }
    }
#endif
    
    // set this to zero to disable collisions with cube sides
#if 0
    if (threadPos.y > 4.0f - params.particleRadius)
    {
        threadPos.y = 4.0f - params.particleRadius;
        threadVel.y *= params.boundaryDamping;
    }
#endif
#if 0
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

    // Check for particles to be removed by height or time
    if ((iters < 2 * params.maxIterations) && ((params.limitParticleLifeByTime && (iters >= params.maxIterations)) ||
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

// calculate address in grid from position (clamping to edges)
__device__ uint calcPreRemovalCellIndex(int3 gridPos, float numIters, float height)
{
    if (numIters > params.maxIterations) {
        return params.numCells;
    }
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
__device__
float W(float r) {
    // Note: r should always be > 0
    float d = 1/1.0; 
    if (r > d) {
        return 0;
    } 
    float secondTerm = d * d - r * r; 
    return (315 / (64 * M_PI * (d * d * d * d * d * d * d * d * d))) * secondTerm * secondTerm * secondTerm; 
}

__device__
float gradW(float r) {
    float d = 1 / 1.0; 
    if (r > d) {
        return 0;
    }
    float secondTerm = d * d - r * r;
    return (315 / (64 * M_PI * (d * d * d * d * d * d * d * d * d))) * -6 * r * secondTerm * secondTerm; 
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
    float h = 2.0 / 1.0;
    if ((2 > h) || (r <= h)) {
        return 32.0 / (M_PI * h * h * h * h * h * h * h * h * h) * (h - r) * (h - r) * (h - r) * r * r * r;
    }
    else if ((r > 0) || (2 * r <= h)) {
        return 32.0 / (M_PI * h * h * h * h * h * h * h * h * h) * (h - r) * (h - r) * (h - r) * r * r * r - (h * h * h * h * h * h) / 64.0;
    } else {
        return 0;
    }
}

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

    // examine neighbouring cells
    float density = 0; 

    // Collide with other Particles
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

    // Collide with other Particles
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
    float mu = 0.0001;
    float mass = 1;
    return mu * mass * (vel2 - vel1) / dens2 * laplacianW(length(pos1 - pos2));
}

__device__
float3 calcPressureForce(float3 pos1, float3 pos2, float dens1, float dens2)
{
    /*float k = 0.005;
    float ro_naught = 20.5;
    float mass = 1;
    float pressure1 = k * (dens1 - ro_naught);
    float pressure2 = k * (dens2 - ro_naught);
    return -1.0 * dir * (mass * (pressure1 + pressure2) / (2 * dens2) * gradW(length(pos1 - pos2)));*/
    float radius = length(pos1 - pos2);
    float3 dir = (pos1 - pos2) / radius;
    float factor=0; 
    if (radius > 2 * params.particleRadius && radius < 4 * params.particleRadius) {
        factor = 200 * (radius - 2 * params.particleRadius) * (radius - 4 * params.particleRadius);
    }
    if (radius > 0 && radius <= 2 * params.particleRadius) {
        factor = 1.0 / radius - 1.0 / (2 * params.particleRadius);
    }
    return dir * factor * 0.01;
}

__device__
float3 calcSurfaceTensionForce(float3 pos1,
                               float3 pos2, 
                               float  norm1, 
                               float  norm2, 
                               float  density1, 
                               float  density2)
{
    float mass1 = 1;
    float mass2 = 1;
    float gamma = 0.1;
    float ro_naught = 20.5;
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

                // collide two spheres
                if (length(particlePos - pos2) < 10 * params.cellSize.x) {
                    particleForce += calcPressureForce(particlePos, pos2, particleDensity, density2);
                    particleForce += calcViscousForce(particlePos, pos2, particleVel, vel2, particleDensity, density2);
                    particleForce += calcSurfaceTensionForce(particlePos, pos2, particleNormal, normal2, particleDensity, density2);
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
              bool   *activeVoxel,
              float4 *voxelPos,
              float  *normals,
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

        float3 forceFromObject = calcForceFromVoxel(voxelGridPos, voxelPos, activeVoxel, particlePos, particleVel, particleForce);
        particleForce += forceFromObject;
    }
#endif

    // collide with cursor sphere
    particleForce += collideSpheres(particlePos, params.colliderPos, particleVel, 
                                    make_float3(0.0f, 0.0f, 0.0f), 
                                    params.particleRadius, params.colliderRadius, 0.0f);

    // Note: Is 1.0 reasonable???
    force[index] = make_float4(particleForce, force[index].w);
}

#endif
