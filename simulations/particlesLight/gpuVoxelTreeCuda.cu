

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
    newBB.lowerBoundary = make_float3(lowerIndex) * sizeOfCell; 
    newBB.upperBoundary = make_float3((lowerIndex + make_uint3(1,1,1))) * sizeOfCell; 
    return newBB; 
}

__device__
bool isOutsideBoundingBox(float3 pos)
{
    float3 lowerBound = boundary.lowerBoundary;
    float3 upperBound = boundary.upperBoundary; 
    if (pos.x < lowerBound.x || pos.y < lowerBound.y || pos.z < lowerBound.z) {
        return true; 
    }
    if (pos.x > upperBound.x || pos.y > upperBound.y || pos.z > upperBound.z) {
        return true;
    }
    return false; 
}


__device__
unsigned int getStatus(float3 pos)
{
    // Start at level 0, offset into cell 0, and the bounding box for the whole gdb
	unsigned int currentLevel = 0;
	BoundingBox currentBB = boundary;
	unsigned int offset = 0; 
    if (isOutsideBoundingBox(pos)) {
        // If outside the bounding box, the voxel is inactive
        return 0.0; 
    }
	while (1) {
        // Otherwise, get the status of the cell we're in
		unsigned int cell = getCell(pos, currentBB, numCellsPerSide[currentLevel]);
		float status = pointersToStatuses[currentLevel][cell + offset];
		// Dig deeper = INF
		if (status != NAN) {
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
void repairVoxelTree(float4 *result,
                     unsigned int *numClaimedInArrayAtLevel,
                     unsigned int numToRepair)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (index >= numToRepair) return;

    float WORK_IN_PROGRESS = INFINITY; 
    float3 pos = make_float3(result[index]); 
    BoundingBox currentBB = boundary; 
    //unsigned int cell; 
    unsigned int offset = 0;
    unsigned int numCellsInLevel = 1; 
    unsigned int cell;
    float amountToReduceStrength = result[index].w;

    for (int level = 0; level < numLevels - 1; ++level) {
        // Get index of cell 
        cell = getCell(pos, currentBB, numCellsPerSide[level]);
        numCellsInLevel *= numCellsPerSide[level] * numCellsPerSide[level] * numCellsPerSide[level]; 
        if (cell + offset >= numCellsInLevel) {
            printf("Problem1 at index %d\n", index);
            return; 
        }
        // Check if work is happening already
        float prevStatus = atomicExch(&(pointersToStatuses[level][cell + offset]), WORK_IN_PROGRESS);
        if (prevStatus != WORK_IN_PROGRESS) {
            // If chunk is not allready dig deeper...
            int chunkNum = pointersToDelimiters[level][cell + offset];
            if (chunkNum == -1) {
                // Get pos to add into
                unsigned int positionToAdd = atomicAdd(&numClaimedInArrayAtLevel[level + 1], 1);
                unsigned int numCellsInChunk = numCellsPerSide[level + 1] * numCellsPerSide[level + 1] * numCellsPerSide[level + 1];

                unsigned int numCellsInNextLevel = numCellsInLevel * numCellsPerSide[level + 1] * numCellsPerSide[level + 1] * numCellsPerSide[level + 1];
                for (int i = 0; i < numCellsInChunk; ++i) {
                    if (positionToAdd * numCellsInChunk + i >= numCellsInNextLevel) {
                       printf("Problem2 at index %d\n", index);
                       return; 
                    }
                    // Set next level to proper values
                    pointersToStatuses[level + 1][positionToAdd * numCellsInChunk + i] = 1;
                    pointersToDelimiters[level + 1][positionToAdd * numCellsInChunk + i] = -1; 
                }
                // Update chunk index
                pointersToDelimiters[level][cell + offset] = positionToAdd; 
            }
            //NAN = dig deeper into tree
            pointersToStatuses[level][cell + offset] = NAN; 
        } else {
            while (pointersToStatuses[level][cell + offset] == WORK_IN_PROGRESS) {}
        }
        unsigned int delimiter = pointersToDelimiters[level][cell + offset];
        unsigned int nextLevelCubeSize = numCellsPerSide[level + 1];
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
    unsigned int maxResultSize = 10000;
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


    unsigned int size; 
    cudaMemcpy(&size, sizeOfResult, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    numThreads = min(256, size); 
    if (numThreads > 0) {
        numBlocks = ceil((float) size / numThreads); 
        if (size > maxResultSize) 
            printf("problem: Size %d is greater than max size %d\n", size, maxResultSize);
        repairVoxelTree<<<numBlocks, numThreads>>>((float4 *) result,
                                                    numClaimedInArrayAtLevel, 
                                                    min(size, maxResultSize));
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