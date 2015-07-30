

#include <math.h>

#include <cuda_runtime.h>
#include "vector_types.h"
#include "helper_math.h"
#include <helper_cuda.h>

#include "gpuVoxelTree.h"

__constant__ unsigned int numLevels;
__constant__ BoundingBox  boundary;
__constant__ unsigned int  numCellsPerSide[10];
__constant__ float voxelSize;
__constant__ float* pointersToStatuses[10];
__constant__ unsigned int* pointersToDelimiters[10]; // Don't need delimiters for the lowest level
__constant__ unsigned int voxelsPerSide; 

// textures for particle position and velocity
//texture<uint, 1, cudaReadModeElementType> voxelStrengthTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

// I don't like #defines, but we can't do static const variables because
//  they have to be available to host and device.  grrr...
#define STATUS_FLAG_WORK_IN_PROGRESS INFINITY
#define STATUS_FLAG_DIG_DEEPER       NAN

// Utility Functions
void getPointersToDeallocateFromGPU(std::vector<void *> statusPointersToDeallocate, 
                                    std::vector<void *> delimiterPointersToDeallocate,
                                    uint numLevels)
{
	checkCudaErrors(cudaMemcpyFromSymbol(&statusPointersToDeallocate[0], pointersToStatuses,
                                         numLevels * sizeof(float *), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyFromSymbol(&delimiterPointersToDeallocate[0], pointersToDelimiters,
                                         numLevels * sizeof(unsigned int *), 0, cudaMemcpyDeviceToHost));
}
void copyDataToConstantMemory(unsigned int numberOfLevels,
                             BoundingBox BB, 
                             std::vector<unsigned int> numberOfCellsPerSide,
                             float sizeOfVoxel,
                             std::vector<void *> pointersToLevelStatuses,
                             std::vector<void *> pointersToLevelDelimiters,
                             unsigned int numberOfVoxelsPerSide)
{
	checkCudaErrors(cudaMemcpyToSymbol(numLevels, (void *) &numberOfLevels, sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(boundary, (void *) &BB, sizeof(BoundingBox)));
	checkCudaErrors(cudaMemcpyToSymbol(numCellsPerSide, (void *) &numberOfCellsPerSide[0], numberOfLevels * sizeof(unsigned int)));
	checkCudaErrors(cudaMemcpyToSymbol(voxelSize, (void *) &sizeOfVoxel, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(pointersToStatuses, (void *) &pointersToLevelStatuses[0], numberOfLevels * sizeof(float *)));
	checkCudaErrors(cudaMemcpyToSymbol(pointersToDelimiters, (void *) &pointersToLevelDelimiters[0], numberOfLevels * sizeof(unsigned int *)));
    checkCudaErrors(cudaMemcpyToSymbol(voxelsPerSide, (void *) &numberOfVoxelsPerSide, sizeof(unsigned int)));
}

// Functions for CUDA
__device__
uint3 calculateCoordsFromIndex(uint index)
{
    // Find the coordinates of a *marching cube* from its index
    uint3 center;
    center.z = index / ((voxelsPerSide + 1) * (voxelsPerSide + 1));
    center.y = (index - center.z * (voxelsPerSide + 1) * (voxelsPerSide + 1)) / (voxelsPerSide + 1); 
    center.x = index - (voxelsPerSide + 1) * (center.y + (voxelsPerSide + 1) * center.z);
    return center;
}


__device__
float3 calculateVoxelCenter(int3 gridPos)
{
    float3 center;
    center.x = boundary.lowerBoundary.x + (voxelSize / 2.0) + gridPos.x * voxelSize;
    center.y = boundary.lowerBoundary.x + (voxelSize / 2.0) + gridPos.y * voxelSize;
    center.z = boundary.lowerBoundary.x + (voxelSize / 2.0) + gridPos.z * voxelSize;
    return center;
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

__device__
unsigned int getCell(float3 pos, BoundingBox boundingBox, unsigned int cubeSize)
{
    // "origin" of box is at the lower boundary
	float3 relPos = pos + -1.0 * boundingBox.lowerBoundary; 
    float sizeOfCell = (boundingBox.upperBoundary.x - boundingBox.lowerBoundary.x) / (float) cubeSize; 
    // Find which cell the position is in
    uint xCoord = (uint) floor(relPos.x / sizeOfCell); 
    uint yCoord = (uint) floor(relPos.y / sizeOfCell); 
    uint zCoord = (uint) floor(relPos.z / sizeOfCell); 
    return zCoord * cubeSize * cubeSize + yCoord * cubeSize + xCoord; 
}

__device__
bool isOutsideBoundingBox(float3 pos, BoundingBox BB)
{
    if (pos.x < BB.lowerBoundary.x || pos.y < BB.lowerBoundary.y || pos.z < BB.lowerBoundary.z) {
        return true; 
    }
    if (pos.x > BB.upperBoundary.x || pos.y > BB.upperBoundary.y || pos.z > BB.upperBoundary.z) {
        return true;
    }
    return false; 
}

__device__
BoundingBox calculateNewBoundingBox(float3 pos, BoundingBox boundingBox, uint cubeSize)
{
    // Find which cell of the old bounding box the pos is in
    float3 offsetFromOrigin = pos + (-1.0f * boundingBox.lowerBoundary);
    float sizeOfCell = (boundingBox.upperBoundary.x - boundingBox.lowerBoundary.x) / (float) cubeSize; 
    uint3 lowerIndex;
    lowerIndex.x = (uint) floor(offsetFromOrigin.x / sizeOfCell);
    lowerIndex.y = (uint) floor(offsetFromOrigin.y / sizeOfCell);
    lowerIndex.z = (uint) floor(offsetFromOrigin.z / sizeOfCell);
    // Calculate the new upper and lower boundaries based on the cell
    BoundingBox newBB; 
    newBB.lowerBoundary = boundingBox.lowerBoundary + make_float3(lowerIndex) * sizeOfCell; 
    newBB.upperBoundary = boundingBox.lowerBoundary + make_float3((lowerIndex + make_uint3(1,1,1))) * sizeOfCell; 
    if (isOutsideBoundingBox(pos, boundingBox)) {
        printf("Problem: Bounding box calculated incorrectly\n");
    }
    return newBB; 
}


__device__
unsigned int getStatus(float3 pos)
{
    // Start at level 0, offset into cell 0, and the bounding box for the whole gdb
	unsigned int currentLevel = 0;
	BoundingBox currentBB = boundary;
	unsigned int offset = 0; 
    if (isOutsideBoundingBox(pos, currentBB)) {
        // If outside the bounding box, the voxel is inactive
        return 0.0; 
    }
	while (1) {
        // Otherwise, get the status of the cell we're in
		unsigned int cell = getCell(pos, currentBB, numCellsPerSide[currentLevel]);
		float status = pointersToStatuses[currentLevel][cell + offset];
		// Dig deeper = INF
		if (status != STATUS_FLAG_DIG_DEEPER) {
            // If it is active or inactive, return the status
			return status;
		} else {
            // Otherwise, find our new offset and bounding box, and loop
			unsigned int delimiter = pointersToDelimiters[currentLevel][cell + offset];
			unsigned int nextLevelCubeSize = numCellsPerSide[currentLevel + 1];
			offset = delimiter * nextLevelCubeSize * nextLevelCubeSize * nextLevelCubeSize; 
			currentBB = calculateNewBoundingBox(pos, currentBB, numCellsPerSide[currentLevel + 1]);
		}

	}
}

__global__
void calculateNewVelocities(float4 *particlePos,
                            float4 *particleVel,
                            float particleRadius,
                            unsigned int numParticles,
                            float deltaTime, 
                            float4 *result,
                            unsigned int *sizeOfResult,
                            unsigned int maxResultSize)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    float3 currentParticlePos = make_float3(particlePos[index]);
    float3 currentParticleVel = make_float3(particleVel[index]);
    float iters = particleVel[index].w; 

    // Loop over all voxels that are touching the particle
    int loopStart = -1.0 * floor(particleRadius / voxelSize);
    int loopEnd = ceil(particleRadius / voxelSize); 

    float3 averagePosition = make_float3(0,0,0); 
    unsigned int numNeighboringVoxels = 0; 

    for (int z = loopStart; z <= loopEnd; ++z) {
    	for (int y = loopStart; y <= loopEnd; ++y) {
    		for (int x = loopStart; x <= loopEnd; ++x) {
    			float3 position = currentParticlePos + voxelSize * make_float3(x, y, z); 
    			float status = getStatus(position); 
    			if (status > 0) {
                    // Get data for average voxel position
    				++numNeighboringVoxels; 
                    averagePosition += position;
                    // Reduce the strength (do so more for glancing blows) 
                    float t_c = 0.1; 
                    float amountToReduceStrength = -10.0 * length(cross(currentParticleVel, currentParticlePos - position)) * deltaTime / t_c;
                    unsigned int indexToAdd = atomicAdd(sizeOfResult, 1);
                    if (indexToAdd < maxResultSize) {
                        // Add position of voxel and amount to reduce strength to our output for later use
                        result[indexToAdd] = make_float4(position, amountToReduceStrength);  
                    }  
    			}
    		}
    	}
    }

    if (numNeighboringVoxels > 0) {
        // get the average position
        averagePosition.x = averagePosition.x / numNeighboringVoxels;
        averagePosition.y = averagePosition.y / numNeighboringVoxels;
        averagePosition.z = averagePosition.z / numNeighboringVoxels;
        
        // The particle reflects around the normal.  
        float3 normalVector = (currentParticlePos - averagePosition) / length(currentParticlePos - averagePosition);
        currentParticleVel -= 2 * dot(normalVector, currentParticleVel) * normalVector;
        currentParticlePos = averagePosition + (2 * particleRadius * normalVector);

        // TODO: Figure out a way to remove particles
    }
    particlePos[index] = make_float4(currentParticlePos, 1.0f);
    particleVel[index] = make_float4(currentParticleVel, iters);
}

__global__
void repairVoxelTree(const float4 *result,
                     const unsigned int numToRepair,
                     unsigned int *numClaimedInArrayAtLevel,
                     unsigned int *addressOfErrorField)
{
    const uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numToRepair) return;

    const float3 pos = make_float3(result[index]); 
    const float amountToReduceStrength = result[index].w;
    BoundingBox currentBB = boundary; 
    unsigned int cell; 
    unsigned int offset = 0;
    unsigned int numCellsInThisLevel = 1; 

    for (unsigned int level = 0; level < numLevels - 1; ++level) {
        // Get index of cell 
        cell = getCell(pos, currentBB, numCellsPerSide[level]);
        numCellsInThisLevel *= numCellsPerSide[level] *
          numCellsPerSide[level] * numCellsPerSide[level]; 
        if (cell + offset >= numCellsInThisLevel) {
            printf("Problem1 at index %d\n", index);
            atomicAdd(addressOfErrorField, unsigned(1));
            return; 
        }
        // First, check if the cell is DIG_DEEPER.  If it is, then we
        //  don't really need to do any fancy logic, we know we can just
        //  go ahead and dig deeper.
        const float firstStatusCheck =
          pointersToStatuses[level][cell + offset];
        if (firstStatusCheck != STATUS_FLAG_DIG_DEEPER) {
            // Now we know that it's either active or work in progress.
            // If it's active, we need to refine the cell.  If it's work
            //  in progress, then we know someone else is refining the cell
            //  and we just wait.
            // Check if work is happening already
            const float secondStatusCheck =
              atomicExch(&(pointersToStatuses[level][cell + offset]),
                         STATUS_FLAG_WORK_IN_PROGRESS);
            if (secondStatusCheck != STATUS_FLAG_WORK_IN_PROGRESS) {
                // If chunk is not allready dig deeper...
                const unsigned int chunkNumber =
                  pointersToDelimiters[level][cell + offset];
                printf("Index %5u is checking chunk number %5u\n",
                       index, chunkNumber);
                if (chunkNumber == INVALID_CHUNK_NUMBER) {
                    printf("Index %5u needs to refine cell on level %2u "
                           "offset %5u cell %3u\n",
                           index, level, offset, cell);
                    // Claim a chunk number at the next level
                    const unsigned int nextLevelsClaimedChunkNumber =
                      atomicAdd(&numClaimedInArrayAtLevel[level + 1], unsigned(1));
                    const unsigned int numCellsInChunkAtNextLevel =
                      numCellsPerSide[level + 1] * numCellsPerSide[level + 1] *
                      numCellsPerSide[level + 1];

                    const unsigned int numCellsInNextLevel =
                      numCellsInThisLevel * numCellsInChunkAtNextLevel;
                    for (unsigned int i = 0; i < numCellsInChunkAtNextLevel; ++i) {
                        if (nextLevelsClaimedChunkNumber * numCellsInChunkAtNextLevel + i >=
                            numCellsInNextLevel) {
                           printf("Index %5u is failing to set next chunk's "
                                  "values\n", index);
                           atomicAdd(addressOfErrorField, unsigned(1));
                           return; 
                        }
                        // Set next level to proper values
                        pointersToStatuses[level + 1][nextLevelsClaimedChunkNumber * numCellsInChunkAtNextLevel + i] = 1;
                        pointersToDelimiters[level + 1][nextLevelsClaimedChunkNumber * numCellsInChunkAtNextLevel + i] = INVALID_CHUNK_NUMBER; 
                    }
                    printf("Index %5u is setting offset of level %3u, cell %3u, offset %3u to %3u\n",
                       index, level, cell, offset, nextLevelsClaimedChunkNumber);
                    // Update chunk index
                    pointersToDelimiters[level][cell + offset] = nextLevelsClaimedChunkNumber; 
                }
                printf("Index %5u is setting status of level %3u, cell %3u, offset %3u to DIG_DEEPER\n",
                       index, level, cell, offset);
                pointersToStatuses[level][cell + offset] = STATUS_FLAG_DIG_DEEPER; 
            } else {
                unsigned int numberOfTimesWeveWaited = 0;
                while (pointersToStatuses[level][cell + offset] ==
                       STATUS_FLAG_WORK_IN_PROGRESS) {
                    ++numberOfTimesWeveWaited;
                    if (numberOfTimesWeveWaited > 100000) {
                      //printf("Index %5u is infinite looping, marking and "
                             //"returning at level %5u\n", index, level);
                      atomicAdd(addressOfErrorField, unsigned(1));
                      return;
                    }
                }
            }
        }
        const unsigned int delimiter = pointersToDelimiters[level][cell + offset];
        const unsigned int nextLevelCubeSize = numCellsPerSide[level + 1];
        offset = delimiter * nextLevelCubeSize * nextLevelCubeSize * nextLevelCubeSize; 
        currentBB = calculateNewBoundingBox(pos, currentBB, numCellsPerSide[level + 1]);
    }
    // TODO: Is amount to Reduce strength negative?
    atomicAdd(&pointersToStatuses[numLevels - 1][cell + offset], amountToReduceStrength);
    return; 	
}

__global__
void coarsenVoxelTree(float4 *result)
{
	return; 
}

void collideWithParticles(float *particlePos,
                          float *particleVel,
                          float  particleRadius,
                          unsigned int numParticles,
                          unsigned int *numClaimedInArrayAtLevel,
                          float deltaTime)
{
	unsigned int numThreads = 256; 
	unsigned int numBlocks = ceil((float) numParticles / numThreads);
    unsigned int maxResultSize = 1000000;
	float *result;
    checkCudaErrors(cudaMalloc((void **) &result, maxResultSize * sizeof(float4))); 
    unsigned int *sizeOfResult; 
    checkCudaErrors(cudaMalloc((void **) &sizeOfResult, 1 * sizeof(unsigned int))); 
    checkCudaErrors(cudaMemset(sizeOfResult, 0, sizeof(unsigned int))); 
    calculateNewVelocities<<<numBlocks, numThreads>>>((float4 *) particlePos,
                                                    (float4 *) particleVel,
                                                    particleRadius,
                                                    numParticles,
                                                    deltaTime,
                                                    (float4 *) result,
                                                    sizeOfResult, 
                                                    maxResultSize);
    getLastCudaError("Kernel execution failed");


    unsigned int numberOfResultsProduced;
    cudaMemcpy(&numberOfResultsProduced, sizeOfResult, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    numThreads = min(256, numberOfResultsProduced);
    if (numThreads > 0) {
        numBlocks = ceil((float) numberOfResultsProduced / numThreads);
        if (numberOfResultsProduced > maxResultSize) {
            fprintf(stderr, "problem: numberOfResultsProduced %d is greater "
                    "than max size %d\n",
                    numberOfResultsProduced, maxResultSize);
            exit(1);
        }
        const unsigned int numberOfResultsToProcess =
          std::min(numberOfResultsProduced, maxResultSize);
        unsigned int *addressOfErrorField;
        checkCudaErrors(cudaMalloc((void **) &addressOfErrorField,
                                   1 * sizeof(unsigned int)));
        checkCudaErrors(cudaMemset(addressOfErrorField, 0,
                                   sizeof(unsigned int)));
        if (numberOfResultsToProcess > 0) {
            printf("     calling repairVoxelTree to process %5u results with "
                   "%4u threads and %5u blocks\n",
                   numberOfResultsToProcess, numThreads, numBlocks);
        }
        repairVoxelTree<<<numBlocks, numThreads>>>((float4 *) result,
                                                   numberOfResultsToProcess,
                                                   numClaimedInArrayAtLevel,
                                                   addressOfErrorField);
        unsigned int numberOfErrors;
        cudaMemcpy(&numberOfErrors, addressOfErrorField, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        if (numberOfResultsToProcess > 0) {
            printf("done calling repairVoxelTree to process %5u results, had "
                   "%5u errors\n", numberOfResultsToProcess, numberOfErrors);
        }
        if (numberOfErrors > 0) {
            fprintf(stderr, "found %u errors after call to repairVoxelTree "
                    "with %u results\n", numberOfErrors,
                    numberOfResultsToProcess);
            exit(1);
        }
    }
    getLastCudaError("Kernel execution failed");

    //coarsenVoxelTree<<<numBlocks, numThreads>>>((float4 *) result);
    cudaFree(result);
    cudaFree(sizeOfResult); 
	
}

__global__
void createMarchingCubesMeshD(float4 *vertexPos,
                              float4 *norm,
                              uint  *tri,
                              uint  *numVerts,
                              uint  *numVerticesClaimed,
                              uint   numVoxelsToDraw)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    // Get gridPos of our grid cube - starts as lower left corner as (0,0,0)
    uint3 gridPos = calculateCoordsFromIndex(index);
    //printf("GridPos: %d, %d, %d numCells: %d voxelSize: %f\n", gridPos.x, gridPos.y, gridPos.z, voxelsPerSide, voxelSize);
    
    // Check if voxels on corner of gridcube are 
    int lookupIndexForActiveVertices = 0;
    float3 cubeVertexPos[8];
    float4 field[8];

    int3 i = make_int3(-1, -1, -1);
    int3 toCheck = make_int3(gridPos) + i;
    cubeVertexPos[0] = calculateVoxelCenter(toCheck);
    bool isActive = getStatus(cubeVertexPos[0]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 0); 
    field[0] = fieldFunc4(cubeVertexPos[0]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 0, lookupIndexForActiveVertices, cubeVertexPos[0].x, cubeVertexPos[0].y, cubeVertexPos[0].z);

    
    i = make_int3(0,-1,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[1] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[1]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 1); 
    field[1] = fieldFunc4(cubeVertexPos[1]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 1, lookupIndexForActiveVertices, cubeVertexPos[1].x, cubeVertexPos[1].y, cubeVertexPos[1].z);

    i = make_int3(0,0,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[2] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[2]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 2); 
    field[2] = fieldFunc4(cubeVertexPos[2]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 2, lookupIndexForActiveVertices, cubeVertexPos[2].x, cubeVertexPos[2].y, cubeVertexPos[2].z);

    i = make_int3(-1,0,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[3] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[3]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 3); 
    field[3] = fieldFunc4(cubeVertexPos[3]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 3, lookupIndexForActiveVertices, cubeVertexPos[3].x, cubeVertexPos[3].y, cubeVertexPos[3].z);

    i = make_int3(-1,-1,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[4] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[4]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 4); 
    field[4] = fieldFunc4(cubeVertexPos[4]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 4, lookupIndexForActiveVertices, cubeVertexPos[4].x, cubeVertexPos[4].y, cubeVertexPos[4].z);

    i = make_int3(0,-1,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[5] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[5]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 5); 
    field[5] = fieldFunc4(cubeVertexPos[5]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 5, lookupIndexForActiveVertices, cubeVertexPos[5].x, cubeVertexPos[5].y, cubeVertexPos[5].z);

    i = make_int3(0,0,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[6] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[6]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 6); 
    field[6] = fieldFunc4(cubeVertexPos[6]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 6, lookupIndexForActiveVertices, cubeVertexPos[6].x, cubeVertexPos[6].y, cubeVertexPos[6].z);

    i = make_int3(-1,0,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[7] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[7]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 7); 
    field[7] = fieldFunc4(cubeVertexPos[7]);
    //printf("Num: %d Lookup: %d Pos: %f, %f, %f\n", 7, lookupIndexForActiveVertices, cubeVertexPos[7].x, cubeVertexPos[7].y, cubeVertexPos[7].z);
    
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

    uint numVerticesToAdd = tex1Dfetch(numVertsTex, lookupIndexForActiveVertices);
    uint positionToAdd = atomicAdd(numVerticesClaimed, numVerticesToAdd); 
    //if (gridPos.z ==0 && index % 1000 == 0) printf("To add: %d Pos: %d Total: %d\n", numVerticesToAdd, positionToAdd, numVoxelsToDraw * 15);
    for (int i= 0; i < numVerticesToAdd; ++i) {

        uint edge = tex1Dfetch(triTex, lookupIndexForActiveVertices*16 + i);
        uint indexToAdd = positionToAdd + i;

        if (indexToAdd < numVoxelsToDraw * 15)
        {
            vertexPos[indexToAdd] = make_float4(vertlist[edge], 1.0f);
            norm[indexToAdd] = make_float4(normList[edge], 0.0f);
        }
    }

}

void generateMarchingCubes(float *pos,
                           float *norm,
                           unsigned int *tri,
                           unsigned int *numVerts,
                           unsigned int *verticesInPosArray,
                           unsigned int numVoxelsToDraw,
                           unsigned int numMarchingCubes)
{
    checkCudaErrors(cudaBindTexture(0, triTex, tri, sizeof(uint) * 256 * 16));
    checkCudaErrors(cudaBindTexture(0, numVertsTex, numVerts, sizeof(uint) * 256));

    // thread per particle
    uint numThreads, numBlocks;
    numThreads = 256;
    numBlocks = ceil((float) numMarchingCubes / (float) numThreads);
    cudaMemset(verticesInPosArray, 0, sizeof(uint));

    // execute the kernel
    //timer->startTimer(5, true);
    createMarchingCubesMeshD<<< numBlocks, numThreads >>>((float4 *) pos,
                                                          (float4 *) norm,
                                                           tri,
                                                           numVerts,
                                                           verticesInPosArray,
                                                           numVoxelsToDraw);
    //timer->stopTimer(5, true);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaUnbindTexture(triTex));
    checkCudaErrors(cudaUnbindTexture(numVertsTex));
}
