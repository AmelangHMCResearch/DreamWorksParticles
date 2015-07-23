#include "voxelObject.h"
#include "particleSystem.cuh"
#include "tables.h"
#include <stack>
#include <algorithm>
#include <stdio.h>


// Taken from Sean Anderson's Bit Twiddling Hacks
uint getNextHighestPowerOfTwo(uint x) {
    --x; 
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4; 
    x |= x >> 8;
    x |= x >> 16;
    ++x; 
    return x; 

}

VoxelObject::VoxelObject(ObjectShape shape, float voxelSize, uint3 cubeSize, float3 origin)
  :  _pos(0),
    _voxelStrength(0),
    _numActiveVoxels(0),
    _maxVoxelStrength(MAX_ROCK_STRENGTH)
{
    _objectParams._voxelSize = voxelSize; // Length of a side of a voxel
    _objectParams._origin = origin;       // Position object is centered at

    if (shape == VOXEL_FROM_FILE) {
        // Get size of sample from file
        /*FILE *fp = fopen("BostonTeapot.raw", "rb");
        if (fp == NULL) {
            printf("Problem opening file 1\n");
            exit(1);
        }
        unsigned short sizeOfObject[3];
        int result = fread((void*) &sizeOfObject[0],sizeof(unsigned short), 3,fp);
        if (result == 0) {
            printf("Problem reading file 1\n");
            exit(1);
        } 
        fclose(fp);*/
        unsigned short sizeOfObject[3] = {256, 256, 178};

        _objectParams._cubeSize.x = getNextHighestPowerOfTwo(sizeOfObject[0]);
        _objectParams._cubeSize.y = getNextHighestPowerOfTwo(sizeOfObject[1]);
        _objectParams._cubeSize.z = getNextHighestPowerOfTwo(sizeOfObject[2]);

        _objectParams._numVoxels = _objectParams._cubeSize.x * _objectParams._cubeSize.y * _objectParams._cubeSize.z;

    } else {
        _objectParams._cubeSize = cubeSize;   // Number of voxels per side
        _objectParams._numVoxels = cubeSize.x * cubeSize.y * cubeSize.z;
    }
    initObject(shape);
}

VoxelObject::~VoxelObject() 
{
    delete [] _pos;
    delete [] _voxelStrength;

    freeArray(_dev_triTable);
    freeArray(_dev_numVertsTable);
    
    unregisterGLBufferObject(_cuda_posvbo_resource);
    glDeleteBuffers(1, (const GLuint *)&_posVBO);

    unregisterGLBufferObject(_cuda_normvbo_resource);
    glDeleteBuffers(1, (const GLuint *)&_normVBO);
}

void VoxelObject::initObject(ObjectShape shape) 
{
    setObjectParameters(&_objectParams);

	unsigned int memSize = _objectParams._numVoxels * sizeof(float);

    _pos = new float[4 * _objectParams._numVoxels];
    _voxelStrength = new float[_objectParams._numVoxels];
    for (int i = 0; i < _objectParams._numVoxels; ++i) {
        _voxelStrength[i] = 0;
    }

    // Allocate active voxel array on GPU
    allocateArray((void **) &_dev_voxelStrength, sizeof(float) * _objectParams._numVoxels);
    allocateArray((void **) &_dev_verticesInPosArray, sizeof(uint));

    // Allocate lookup tables for marching cubes on the GPU
    allocateArray((void **) &_dev_triTable, sizeof(uint) * 256 * 16);
    cudaMemcpy(_dev_triTable, triTable, sizeof(uint) * 256 * 16, cudaMemcpyHostToDevice);
    allocateArray((void **) &_dev_numVertsTable, sizeof(uint) * 256);
    cudaMemcpy(_dev_numVertsTable, numVertsTable, sizeof(uint) * 256, cudaMemcpyHostToDevice);

    // Create the VBO
    uint numMarchingCubes = (_objectParams._cubeSize.x + 1) * (_objectParams._cubeSize.y + 1) * (_objectParams._cubeSize.z + 1);
    _objectParams._maxVoxelsToDraw = std::min(numMarchingCubes, (uint) 128 * 128 * 128);
    uint bufferSize = _objectParams._maxVoxelsToDraw * 4 * 15 * sizeof(float); 
    glGenBuffers(1, &_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);

    glGenBuffers(1, &_normVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _normVBO);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_normVBO, &_cuda_normvbo_resource);
 
 

    initShape(shape);

}

void VoxelObject::generateLandscapeStrength()
{
    // Generate rocks randomly, where lower levels = more likely to be rocks
    for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
    {
        for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
        {
            for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
            {
                unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;
                int isRock = (rand() % _objectParams._cubeSize.y) > y;
                if (isRock) {
                    _voxelStrength[i] = _maxVoxelStrength;
                } else {
                    _voxelStrength[i] = _maxVoxelStrength * 0.5; 
                }
            }
        }
    }
    // Generate softer areas using depth-first search
    std::stack<uint3> voxelsToVisit;
    uint3 startingVoxel = make_uint3(_objectParams._cubeSize.x * 0.5,
                                      _objectParams._cubeSize.y - 1,
                                      _objectParams._cubeSize.z * 0.5); 
    voxelsToVisit.push(startingVoxel);

    while (!voxelsToVisit.empty()) {
        uint3 currentVoxel = voxelsToVisit.top();
        voxelsToVisit.pop();
        uint currentIndex = getVoxelIndex(currentVoxel);
        if (_voxelStrength[currentIndex] != _maxVoxelStrength / 4.0) {
            _voxelStrength[currentIndex] = _maxVoxelStrength / 4.0;
            // Add all surrounding voxels to queue
            // Unless they are a rock 
            for (int z = -1; z <= 1; ++z) {
                for (int x = -1; x <= 1; ++x) {
                    for (int y = -1; y <= 1; ++y) {
                        if (((y == 0 && z == 0) || (x == 0 && z == 0) || (x == 0 && y == 0)) && !(x == 0 && y == 0 && z == 0)) {
                            uint3 neighborVoxel = make_uint3(currentVoxel.x + x, currentVoxel.y + y, currentVoxel.z + z);
                            uint neighborIndex = getVoxelIndex(neighborVoxel);
                            if (voxelIsInBounds(neighborVoxel)) {
                                if (_voxelStrength[neighborIndex] != _maxVoxelStrength && rand() % 3 != 0) {
                                    voxelsToVisit.push(neighborVoxel);
                                } 
                            } 
                            else if (neighborVoxel.x > _objectParams._cubeSize.x || neighborVoxel.x < 0) {
                                // End the search if we hit either end of the area
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

}

void VoxelObject::initShape(ObjectShape shape)
{
	srand(1973);
    switch (shape)
    {
        default:
        case VOXEL_CUBE:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            _voxelStrength[i] = _maxVoxelStrength;
                            ++_numActiveVoxels; 
                        }
                    }
                }
            }
        }
        break;

        case VOXEL_GEOLOGY:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            ++_numActiveVoxels; 
                            _voxelStrength[i] = _maxVoxelStrength / 4.0;
                        }
                    }
                }
            }
            //generateLandscapeStrength();
        }
        break;

        case VOXEL_PLANE:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;
                        if (i < _objectParams._numVoxels)
                        {
                            if (y == 0) {
                                _voxelStrength[i] = _maxVoxelStrength;
                                ++_numActiveVoxels; 
                            }
                        }
                    }
                }
            }
        }
        break;
        case VOXEL_SPHERE:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;

                        float xPos = _objectParams._origin.x + (_objectParams._voxelSize / 2.0) + (x - _objectParams._cubeSize.x / 2.0) * _objectParams._voxelSize;
                        float yPos = _objectParams._origin.y + (_objectParams._voxelSize / 2.0) + (y - _objectParams._cubeSize.y / 2.0) *_objectParams. _voxelSize;
                        float zPos = _objectParams._origin.z + (_objectParams._voxelSize / 2.0) + (z - _objectParams._cubeSize.z / 2.0) * _objectParams._voxelSize;
                        float radius = sqrt((xPos - _objectParams._origin.x) * (xPos - _objectParams._origin.x) + 
                                            (yPos - _objectParams._origin.y) * (yPos - _objectParams._origin.y) +
                                            (zPos - _objectParams._origin.z) * (zPos - _objectParams._origin.z));
                        if (radius <= (_objectParams._cubeSize.x * _objectParams._voxelSize) / 2.0) {
                            _voxelStrength[i] = _maxVoxelStrength;
                            ++_numActiveVoxels; 
                        } 
                    }
                }
            }
        }
        break;
        case VOXEL_FROM_FILE:
        {
            FILE *fp = fopen("BostonTeapot.raw", "rb");
            if (fp == NULL) {
                printf("Problem opening file 2\n");
                exit(1);
            }

            /*unsigned short dataSize[3];
            int result = fread((void*) &dataSize[0],sizeof(unsigned short), 3,fp);
            if (result == 0) {
                printf("Problem reading file 2\n");
                exit(1); 
            }*/
            unsigned short dataSize[3] = {256, 256, 178};    

            for (unsigned int z = 0; z < _objectParams._cubeSize.z; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize.y; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize.x; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize.x * _objectParams._cubeSize.y) + (y * _objectParams._cubeSize.x) + x;
                        if (x < dataSize[0] && y < dataSize[1] && z < dataSize[2]) {
                            unsigned short strength; 
                            int result = fread(&strength, sizeof(char), 1, fp);
                            if (result == 0 && (feof(fp) || ferror(fp))) {
                               printf("Problem reading file 3 at index %d\n", i);
                               exit(1); 
                            }
                            if (strength < 60) {
                                _voxelStrength[i] = 0; 
                            } else {
                                _voxelStrength[i] = _maxVoxelStrength; 
                            }
                            
                        } else {
                            _voxelStrength[i] = 0; 
                        }
                    }
                }
            }
            fclose(fp); 
        }
        break;
    }
    cudaMemcpy(_dev_voxelStrength, _voxelStrength, sizeof(int) * _objectParams._numVoxels, cudaMemcpyHostToDevice);

}

bool VoxelObject::voxelIsInBounds(uint3 gridPos)
{
    if (gridPos.x >= 0 && gridPos.x < _objectParams._cubeSize.x) {
        if (gridPos.y >= 0 && gridPos.y < _objectParams._cubeSize.y) {
            if (gridPos.z >= 0 && gridPos.z < _objectParams._cubeSize.z) {
                return true;
            }
        }
    }

    return false;
}

// Warning: Should only be called on an in-bounds voxel
uint VoxelObject::getVoxelIndex(uint3 gridPos)
{
    return gridPos.z * _objectParams._cubeSize.x * _objectParams._cubeSize.y +
           gridPos.y * _objectParams._cubeSize.x + gridPos.x; 
}

float* VoxelObject::getPosArray() {
    float *dPos = (float *) mapGLBufferObject(&_cuda_posvbo_resource);
    return dPos;
}

float* VoxelObject::getNormalArray() {
    float *dNorm = (float *) mapGLBufferObject(&_cuda_normvbo_resource);
    return dNorm;
}

float* VoxelObject::getCpuPosArray() {
    return _pos;
}

float* VoxelObject::getVoxelStrengthFromGPU() {
    cudaMemcpy(_voxelStrength, _dev_voxelStrength, sizeof(float) * _objectParams._numVoxels, cudaMemcpyDeviceToHost);
    return _voxelStrength;
}


void VoxelObject::unbindPosArray() {
    unmapGLBufferObject(_cuda_posvbo_resource);
}

void VoxelObject::unbindNormalArray() {
    unmapGLBufferObject(_cuda_normvbo_resource);
}
