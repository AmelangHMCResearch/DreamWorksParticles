

#include <math.h>
#include <limits.h>

#include <cuda_runtime.h>
#include "vector_types.h"
#include "helper_math.h"
#include <helper_cuda.h>

#include "gpuVoxelTree.h"

__constant__ unsigned int numLevels;
__constant__ BoundingBox  boundary;
__constant__ unsigned int  numCellsPerSide[10];
__constant__ float voxelSize;
__constant__ volatile float* pointersToStatuses[10];
__constant__ volatile unsigned int* pointersToDelimiters[10]; // Don't need delimiters for the lowest level
__constant__ unsigned int voxelsPerSide; 

// textures for particle position and velocity
//texture<unsigned int, 1, cudaReadModeElementType> voxelStrengthTex;
texture<unsigned int, 1, cudaReadModeElementType> triTex;
texture<unsigned int, 1, cudaReadModeElementType> numVertsTex;      

// Utility Functions
void getPointersToDeallocateFromGPU(const unsigned int numberOfLevels,
                                    std::vector<void *> * statusPointersToDeallocate, 
                                    std::vector<void *> * delimiterPointersToDeallocate)
{
    checkCudaErrors(cudaMemcpyFromSymbol(&statusPointersToDeallocate->at(0), pointersToStatuses,
                                         numberOfLevels * sizeof(float *), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyFromSymbol(&delimiterPointersToDeallocate->at(0), pointersToDelimiters,
                                         numberOfLevels * sizeof(unsigned int *), 0, cudaMemcpyDeviceToHost));
}
void copyDataToConstantMemory(const unsigned int numberOfLevels,
                              const BoundingBox BB,
                              const std::vector<unsigned int> & numberOfCellsPerSide,
                              const float sizeOfVoxel,
                              const std::vector<void *> & pointersToLevelStatuses,
                              const std::vector<void *> & pointersToLevelDelimiters,
                              const unsigned int numberOfVoxelsPerSide)
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
uint3 calculateCoordsFromIndex(unsigned int index)
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
    unsigned int xCoord = min((unsigned int) max(floor(relPos.x / sizeOfCell), 0.0f), cubeSize - 1); 
    unsigned int yCoord = min((unsigned int) max(floor(relPos.y / sizeOfCell), 0.0f), cubeSize - 1); 
    unsigned int zCoord = min((unsigned int) max(floor(relPos.z / sizeOfCell), 0.0f), cubeSize - 1); 
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
BoundingBox calculateNewBoundingBox(float3 pos, BoundingBox boundingBox, unsigned int cubeSize)
{
    // Find which cell of the old bounding box the pos is in
    float3 offsetFromOrigin = pos + (-1.0f * boundingBox.lowerBoundary);
    float sizeOfCell = (boundingBox.upperBoundary.x - boundingBox.lowerBoundary.x) / (float) cubeSize; 
    uint3 lowerIndex;
    lowerIndex.x = min((unsigned int) max(floor(offsetFromOrigin.x / sizeOfCell), 0.0f), cubeSize - 1);
    lowerIndex.y = min((unsigned int) max(floor(offsetFromOrigin.y / sizeOfCell), 0.0f), cubeSize - 1);
    lowerIndex.z = min((unsigned int) max(floor(offsetFromOrigin.z / sizeOfCell), 0.0f), cubeSize -1);
    // Calculate the new upper and lower boundaries based on the cell
    BoundingBox newBB; 
    newBB.lowerBoundary = boundingBox.lowerBoundary + make_float3(lowerIndex) * sizeOfCell; 
    newBB.upperBoundary = boundingBox.lowerBoundary + make_float3((lowerIndex + make_uint3(1,1,1))) * sizeOfCell; 
    if (isOutsideBoundingBox(pos, boundingBox) && (offsetFromOrigin.x > 0) && (offsetFromOrigin.y > 0) && (offsetFromOrigin.z > 0)) {
        printf("Problem: Bounding box calculated incorrectly.Pos: %f, %f, %f Box: (%f, %f, %f), (%f, %f, %f) \n", pos.x, pos.y, pos.z, newBB.lowerBoundary.x, newBB.lowerBoundary.y, newBB.lowerBoundary.z, newBB.upperBoundary.x, newBB.upperBoundary.y, newBB.upperBoundary.z);
    }
    return newBB; 
}


__device__
float getStatus(float3 pos)
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
            //printf("Found status %f position %f, %f, %f\n", status, pos.x, pos.y, pos.z);
            // If it is active or inactive, return the status
			return status;
		} else {
            // Otherwise, find our new offset and bounding box, and loop
			unsigned int delimiter = pointersToDelimiters[currentLevel][cell + offset];
			unsigned int nextLevelCubeSize = numCellsPerSide[currentLevel + 1];
			offset = delimiter * nextLevelCubeSize * nextLevelCubeSize * nextLevelCubeSize; 
			currentBB = calculateNewBoundingBox(pos, currentBB, numCellsPerSide[currentLevel]);
            ++currentLevel; 
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
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    float3 currentParticlePos = make_float3(particlePos[index]);
    float3 currentParticleVel = make_float3(particleVel[index]);
    float iters = particleVel[index].w; 

    // Loop over all voxels that are touching the particle
    int loopEnd = ceil(particleRadius / voxelSize);
    int loopStart = -1.0 * ceil(particleRadius / voxelSize); 

    float3 averagePosition = make_float3(0,0,0); 
    unsigned int numNeighboringVoxels = 0; 

    for (int z = loopStart; z <= loopEnd; ++z) {
    	for (int y = loopStart; y <= loopEnd; ++y) {
    		for (int x = loopStart; x <= loopEnd; ++x) {
    			float3 position = currentParticlePos + voxelSize * make_float3(x, y, z); 
    			float status = getStatus(position); 
    			if (status > 0.0f) {
                    // Get data for average voxel position
    				++numNeighboringVoxels; 
                    averagePosition += position;
                    // Reduce the strength (do so more for glancing blows) 
                    float t_c = 0.1; 
                    float amountToReduceStrength = -10.0 * length(cross(currentParticleVel, currentParticlePos - position)) * deltaTime / t_c;
                    unsigned int indexToAdd = atomicAdd(sizeOfResult, 1);
                    if (indexToAdd < maxResultSize) {
                        //printf("Repairing: Status: %f Pos: %f, %f, %f\n", status, position.x, position.y, position.z);
                        // Add position of voxel and amount to reduce strength to our output for later use
                        result[indexToAdd] = make_float4(position, amountToReduceStrength);  
                    }
    			}
    		}
    	}
    }

    if (numNeighboringVoxels > 0) {
        //printf("Index: %u NumNeighbors: %u\n", index, numNeighboringVoxels);
        // get the average position
        averagePosition.x = averagePosition.x / numNeighboringVoxels;
        averagePosition.y = averagePosition.y / numNeighboringVoxels;
        averagePosition.z = averagePosition.z / numNeighboringVoxels;
        
        // The particle reflects around the normal.  
        float3 normalVector = (currentParticlePos - averagePosition) / length(currentParticlePos - averagePosition);
        currentParticleVel -= 2 * dot(normalVector, currentParticleVel) * normalVector;
        currentParticlePos = averagePosition + (2.0 * particleRadius * normalVector);

        // TODO: Figure out a way to remove particles
    }
    particlePos[index] = make_float4(currentParticlePos, 1.0f);
    particleVel[index] = make_float4(currentParticleVel, iters);
}




__global__
void repairVoxelTree(const float4 *result,
                     const unsigned int numberOfResults,
                     unsigned int *numClaimedInArrayAtLevel,
                     unsigned int *addressOfErrorField)
{
    const unsigned int resultIndex = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (resultIndex >= numberOfResults) return;

    const float3 resultPosition = make_float3(result[resultIndex]);
    const float amountToReduceStrength = result[resultIndex].w;
    BoundingBox currentBB = boundary;
    unsigned int cell;
    unsigned int offset = 0;
    unsigned int numCellsInThisLevel = 1;

    /*
    printf("thread starting with block %4u thread %4u, resultIndex %5u/%5u, %2u levels\n",
           blockIdx.x, threadIdx.x, resultIndex, numberOfResults, numLevels);
    */

    for (unsigned int level = 0; level < numLevels - 1; ++level) {
        /*
        printf("block %4u thread %4u, resultIndex %5u/%5u, level %2u/%2u\n",
               blockIdx.x, threadIdx.x, resultIndex, numberOfResults, level, numLevels);
        */

        // Get index of cell
        cell = getCell(resultPosition, currentBB, numCellsPerSide[level]);
        numCellsInThisLevel *= numCellsPerSide[level] *
          numCellsPerSide[level] * numCellsPerSide[level];
        if (cell + offset >= numCellsInThisLevel) {
            printf("Problem1 at index %d. Pos: %f, %f, %f cellsPerSide: %u Cell: %u offset: %u level: %u\n", resultIndex, resultPosition.x, resultPosition.y, resultPosition.z, numCellsPerSide[level],cell, offset, level);
            atomicAdd(addressOfErrorField, unsigned(1));
            return;
        }
        // First, check if the cell is DIG_DEEPER.  If it is, then we
        //  don't really need to do any fancy logic, we know we can just
        //  go ahead and dig deeper.
        const float firstStatusCheck =
          pointersToStatuses[level][cell + offset];
        if (firstStatusCheck != STATUS_FLAG_DIG_DEEPER) {
            // Now we know that the we weren't dig deeper when we first tried.
            // That means we could be active, work in progress, or maybe someone
            //  is working on it right now and it'll be dig deeper by our next
            //  check.
            // If it's active, we need to refine the cell.
            // If it's work in progress, then we know someone else is
            //  refining the cell and we just wait.
            // If it's dig deeper, we just keep going.
            const float secondStatusCheck =
              atomicExch((float*)&(pointersToStatuses[level][cell + offset]),
                         STATUS_FLAG_WORK_IN_PROGRESS);
            bool thisThreadCanContinue = false;
            unsigned int numberOfTimesWeveSpunOnWorkInProgress = 0;
            while (thisThreadCanContinue == false) {
                if (secondStatusCheck == STATUS_FLAG_DIG_DEEPER) {
                    // set the status
                    atomicExch((float*)&(pointersToStatuses[level][cell + offset]),
                               STATUS_FLAG_DIG_DEEPER);
                    thisThreadCanContinue = true;
                } else if (secondStatusCheck == STATUS_FLAG_WORK_IN_PROGRESS) {
                    const float newStatus =
                      pointersToStatuses[level][cell + offset];
                    if (newStatus == STATUS_FLAG_DIG_DEEPER) {
                        /*
                          // the results of this printf seem to be so
                          //  wrong; i don't know what's wrong.
                        printf("resultIndex %5u/%5u is done waiting on work in progress "
                               "at level %2u/%2u, offset %5u, cell %5u, ",
                               "iteration %5u\n",
                               resultIndex, numberOfResults, level,
                               numLevels, offset, cell,
                               numberOfTimesWeveSpunOnWorkInProgress);
                        */
                        // we're ready to go!
                        thisThreadCanContinue = true;
                    }
                    ++numberOfTimesWeveSpunOnWorkInProgress;
                    if (numberOfTimesWeveSpunOnWorkInProgress > 1000000) {
                        printf("resultIndex %5u/%5u is infinite looping "
                               "at level %2u/%2u, offset %5u, cell %5u\n",
                               resultIndex, numberOfResults, level,
                               numLevels, offset, cell);
                        atomicAdd(addressOfErrorField, unsigned(1));
                        return;
                    }
                } else {
                    // It must be active
                    /*
                    printf("Status seen by resultIndex %5u at level %2u, cell %5u offset "
                           "%5u is %8f (active)\n", resultIndex, level, cell, offset, secondStatusCheck);
                    */
                    const unsigned int chunkNumber =
                      pointersToDelimiters[level][cell + offset];
                    if (chunkNumber != INVALID_CHUNK_NUMBER) {
                        printf("error, resultIndex %5u found a status of %8f which is active, but the chunk number of %5u was valid, which shouldn't happen.\n",resultIndex, secondStatusCheck, chunkNumber);
                        atomicAdd(addressOfErrorField, unsigned(1));
                        return;
                    }
                    // By now, the chunkNumber must be INVALID_CHUNK_NUMBER.
                    // Claim a chunk number at the next level.
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
                           printf("resultIndex %5u is failing to set next chunk's "
                                  "values\n", resultIndex);
                           atomicAdd(addressOfErrorField, unsigned(1));
                           return;
                        }
                        // Set next level to proper initial values of active, with
                        //  no chunk number
                        const unsigned int indexOfValue =
                          nextLevelsClaimedChunkNumber * numCellsInChunkAtNextLevel + i;
                        atomicExch((float*)&(pointersToStatuses[level + 1][indexOfValue]),
                                   0.0001);
                        atomicExch((unsigned int*)&(pointersToDelimiters[level + 1][indexOfValue]),
                                   INVALID_CHUNK_NUMBER);
                    }
                    // Update chunk index
                    atomicExch((unsigned int*)&(pointersToDelimiters[level][cell + offset]),
                               nextLevelsClaimedChunkNumber);

                    threadfence();

                    // set the status
                    atomicExch((float*)&(pointersToStatuses[level][cell + offset]),
                               STATUS_FLAG_DIG_DEEPER);
                    /*
                    printf("resultIndex %5u set status/offset of level %2u, cell %5u, offset %5u to %8f/%5u\n",
                           resultIndex, level, cell, offset,
                           pointersToStatuses[level][cell + offset],
                           pointersToDelimiters[level][cell + offset]);
                    */
                    thisThreadCanContinue = true;
                }
            }
        }
        const unsigned int delimiter = pointersToDelimiters[level][cell + offset];
        const unsigned int nextLevelCubeSize = numCellsPerSide[level + 1];
        offset = delimiter * nextLevelCubeSize * nextLevelCubeSize * nextLevelCubeSize; 
        currentBB = calculateNewBoundingBox(resultPosition, currentBB, numCellsPerSide[level]);
    }
    //if (*addressOfErrorField > 0) printf("Num errors: %u\n", *addressOfErrorField);
    // TODO: Is amount to Reduce strength negative?
    cell = getCell(resultPosition, currentBB, numCellsPerSide[numLevels - 1]);
    atomicAdd((float*)&pointersToStatuses[numLevels - 1][cell + offset],
              amountToReduceStrength);
    return;
}

void createShape(const float *result,
                 const unsigned int numberOfResults,
                 unsigned int *numClaimedInArrayAtLevel,
                 unsigned int *addressOfErrorField)
{
    unsigned int numThreads = 256; 
    unsigned int numBlocks = ceil((float) numberOfResults / numThreads);
    repairVoxelTree<<<numBlocks, numThreads>>>((float4 *) result,
                                                numberOfResults,
                                                numClaimedInArrayAtLevel,
                                                addressOfErrorField);
    getLastCudaError("Kernel execution failed");
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
    unsigned int zero = 0;
    checkCudaErrors(cudaMemcpy(sizeOfResult, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice)); 
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
        checkCudaErrors(cudaMemcpy(addressOfErrorField, &zero,
                                   sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (numberOfResultsToProcess > 0) {
            //printf("results = %u\n", numberOfResultsProduced);
            /*printf("calling repairVoxelTree to process %4u results with "
                   "%4u threads and %4u blocks\n",
                   numberOfResultsToProcess, numThreads, numBlocks);*/
        }
        repairVoxelTree<<<numBlocks, numThreads>>>((float4 *) result,
                                                   numberOfResultsToProcess,
                                                   numClaimedInArrayAtLevel,
                                                   addressOfErrorField);
        unsigned int numberOfErrors;
        cudaMemcpy(&numberOfErrors, addressOfErrorField, sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        if (numberOfResultsToProcess > 0) {
            /*fprintf(stderr, "found %u errors after call to repairVoxelTree "
                    "with %u results\n", numberOfErrors,
                    numberOfResultsToProcess);*/
            //exit(1);
        }
        if (numberOfErrors > 0) {
            printf("Exited with %u errors\n", numberOfErrors);
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
                              unsigned int  *tri,
                              unsigned int  *numVerts,
                              unsigned int  *numVerticesClaimed,
                              unsigned int   numVoxelsToDraw)
{
    unsigned int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    // Get gridPos of our grid cube - starts as lower left corner as (0,0,0)
    uint3 gridPos = calculateCoordsFromIndex(index);
    
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

    
    i = make_int3(0,-1,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[1] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[1]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 1); 
    field[1] = fieldFunc4(cubeVertexPos[1]);

    i = make_int3(0,0,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[2] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[2]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 2); 
    field[2] = fieldFunc4(cubeVertexPos[2]);

    i = make_int3(-1,0,-1);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[3] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[3]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 3); 
    field[3] = fieldFunc4(cubeVertexPos[3]);

    i = make_int3(-1,-1,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[4] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[4]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 4); 
    field[4] = fieldFunc4(cubeVertexPos[4]);

    i = make_int3(0,-1,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[5] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[5]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 5); 
    field[5] = fieldFunc4(cubeVertexPos[5]);

    i = make_int3(0,0,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[6] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[6]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 6); 
    field[6] = fieldFunc4(cubeVertexPos[6]);

    i = make_int3(-1,0,0);
    toCheck = make_int3(gridPos) + i;
    cubeVertexPos[7] = calculateVoxelCenter(toCheck);
    isActive = getStatus(cubeVertexPos[7]) > 0;
    lookupIndexForActiveVertices = lookupIndexForActiveVertices | (isActive << 7); 
    field[7] = fieldFunc4(cubeVertexPos[7]);
    
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

    unsigned int numVerticesToAdd = tex1Dfetch(numVertsTex, lookupIndexForActiveVertices);
    unsigned int positionToAdd = atomicAdd(numVerticesClaimed, numVerticesToAdd); 
    //if (gridPos.z ==0 && index % 1000 == 0) printf("To add: %d Pos: %d Total: %d\n", numVerticesToAdd, positionToAdd, numVoxelsToDraw * 15);
    for (int i= 0; i < numVerticesToAdd; ++i) {

        unsigned int edge = tex1Dfetch(triTex, lookupIndexForActiveVertices*16 + i);
        unsigned int indexToAdd = positionToAdd + i;

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
    checkCudaErrors(cudaBindTexture(0, triTex, tri, sizeof(unsigned int) * 256 * 16));
    checkCudaErrors(cudaBindTexture(0, numVertsTex, numVerts, sizeof(unsigned int) * 256));

    // thread per particle
    unsigned int numThreads, numBlocks;
    numThreads = 256;
    numBlocks = ceil((float) numMarchingCubes / (float) numThreads);
    unsigned int zero = 0; 
    cudaMemcpy(verticesInPosArray, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // execute the kernel
    createMarchingCubesMeshD<<< numBlocks, numThreads >>>((float4 *) pos,
                                                          (float4 *) norm,
                                                           tri,
                                                           numVerts,
                                                           verticesInPosArray,
                                                           numVoxelsToDraw);

    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaUnbindTexture(triTex));
    checkCudaErrors(cudaUnbindTexture(numVertsTex));
}
