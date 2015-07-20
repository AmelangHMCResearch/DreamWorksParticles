#include "voxelObject.h"
#include "particleSystem.cuh"
#include <stack>

VoxelObject::VoxelObject(ObjectShape shape, float voxelSize, uint3 cubeSize, float3 origin)
  :  _pos(0),
    _voxelStrength(0),
    _numActiveVoxels(0),
    _maxVoxelStrength(MAX_ROCK_STRENGTH)
{
    _objectParams._voxelSize = voxelSize; // Length of a side of a voxel
    _objectParams._cubeSize = cubeSize;   // Number of voxels per side
    _objectParams._numVoxels = cubeSize.x * cubeSize.y * cubeSize.z;
    _objectParams._origin = origin;       // Position object is centered at
    initObject(shape);
}

VoxelObject::~VoxelObject() 
{
    delete [] _pos;
    delete [] _voxelStrength;

    freeArray(_dev_voxelStrength);

    unregisterGLBufferObject(_cuda_colorvbo_resource);
    unregisterGLBufferObject(_cuda_posvbo_resource);
    glDeleteBuffers(1, (const GLuint *)&_posVBO);
    glDeleteBuffers(1, (const GLuint *)&_colorVBO);
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

    // Create the VBO
    glGenBuffers(1, &_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferData(GL_ARRAY_BUFFER, _objectParams._numVoxels * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);
 

    initShape(shape);

    // Create the color buff
    _colorVBO = createVBO(_objectParams._numVoxels * 4 * sizeof(float));
    registerGLBufferObject(_colorVBO, &_cuda_colorvbo_resource);
    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, _colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for (unsigned int i = 0; i < _objectParams._numVoxels; i++)
    {
        *ptr++ = 0.0;
        *ptr++ = 0.0;
        *ptr++ = 0.0;
        *ptr++ = 1.0;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);

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
                        unsigned int i = (z*_objectParams._cubeSize.z * _objectParams._cubeSize.z) + (y * _objectParams._cubeSize.y) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            _voxelStrength[i] = _maxVoxelStrength;
                            ++_numActiveVoxels; 
                            // Calculate center of voxels for use in VBO rendering
                            _pos[i*4] = _objectParams._origin.x + (_objectParams._voxelSize / 2.0) + (x - _objectParams._cubeSize.x / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+1] = _objectParams._origin.y + (_objectParams._voxelSize / 2.0) + (y - _objectParams._cubeSize.y / 2.0) *_objectParams. _voxelSize;
                            _pos[i*4+2] = _objectParams._origin.z + (_objectParams._voxelSize / 2.0) + (z - _objectParams._cubeSize.z / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+3] = 1.0f;
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
                            // Calculate center of voxels for use in VBO rendering
                            _pos[i*4] = _objectParams._origin.x + (_objectParams._voxelSize / 2.0) + (x - _objectParams._cubeSize.x / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+1] = _objectParams._origin.y + (_objectParams._voxelSize / 2.0) + (y - _objectParams._cubeSize.y / 2.0) *_objectParams. _voxelSize;
                            _pos[i*4+2] = _objectParams._origin.z + (_objectParams._voxelSize / 2.0) + (z - _objectParams._cubeSize.z / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+3] = 1.0f;
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
                        unsigned int i = (z*_objectParams._cubeSize.z * _objectParams._cubeSize.z) + (y * _objectParams._cubeSize.y) + x;
                        if (i < _objectParams._numVoxels)
                        {
                            if (y == 0) {
                                _voxelStrength[i] = _maxVoxelStrength;
                                ++_numActiveVoxels; 
                                // Calculate center of voxels for use in VBO rendering
                                _pos[i*4] = _objectParams._origin.x + (_objectParams._voxelSize / 2.0) + (x - _objectParams._cubeSize.x / 2.0) * _objectParams._voxelSize;
                                _pos[i*4+1] = _objectParams._origin.y + (_objectParams._voxelSize / 2.0) + (y - _objectParams._cubeSize.y / 2.0) *_objectParams. _voxelSize;
                                _pos[i*4+2] = _objectParams._origin.z + (_objectParams._voxelSize / 2.0) + (z - _objectParams._cubeSize.z / 2.0) * _objectParams._voxelSize;
                                _pos[i*4+3] = 1.0f;
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
                        unsigned int i = (z*_objectParams._cubeSize.z * _objectParams._cubeSize.z) + (y * _objectParams._cubeSize.y) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            float xPos = _objectParams._origin.x + (_objectParams._voxelSize / 2.0) + (x - _objectParams._cubeSize.x / 2.0) * _objectParams._voxelSize;
                            float yPos = _objectParams._origin.y + (_objectParams._voxelSize / 2.0) + (y - _objectParams._cubeSize.y / 2.0) *_objectParams. _voxelSize;
                            float zPos = _objectParams._origin.z + (_objectParams._voxelSize / 2.0) + (z - _objectParams._cubeSize.z / 2.0) * _objectParams._voxelSize;
                            float radius = sqrt((xPos - _objectParams._origin.x) * (xPos - _objectParams._origin.x) + 
                                                (yPos - _objectParams._origin.y) * (yPos - _objectParams._origin.y) +
                                                (zPos - _objectParams._origin.z) * (zPos - _objectParams._origin.z));
                            if (radius <= (_objectParams._cubeSize.x * _objectParams._voxelSize) / 2.0) {
                                _voxelStrength[i] = _maxVoxelStrength;
                                ++_numActiveVoxels; 
                                // Calculate center of voxels for use in VBO rendering
                                _pos[i*4] = xPos;
                                _pos[i*4+1] = yPos;
                                _pos[i*4+2] = zPos;
                                _pos[i*4+3] = 1.0f;
                            } 
                        }
                    }
                }
            }
        }
        break;
    }
    cudaMemcpy(_dev_voxelStrength, _voxelStrength, sizeof(int) * _objectParams._numVoxels, cudaMemcpyHostToDevice);

    // Copy position data to vbo
    unregisterGLBufferObject(_cuda_posvbo_resource);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _objectParams._numVoxels*4*sizeof(float), _pos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);

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

unsigned int
VoxelObject::createVBO(unsigned int size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}
