#include "voxelObject.h"
#include "particleSystem.cuh"

VoxelObject::VoxelObject(ObjectShape shape, float voxelSize, unsigned int cubeSize, float3 origin)
  :  _pos(0),
    _activeVoxel(0)
{
    _objectParams._voxelSize = voxelSize;
    _objectParams._cubeSize = cubeSize; 
    _objectParams._numVoxels = cubeSize * cubeSize * cubeSize;
    _objectParams._origin = origin;
    initObject(shape);
}

VoxelObject::~VoxelObject() 
{
    delete [] _pos;

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
    _activeVoxel = new bool[_objectParams._numVoxels];

    // Allocate active voxel array on GPU
    allocateArray((void **) &_dev_activeVoxel, memSize);

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
        *ptr++ = 1.0;
        *ptr++ = 0.0;
        *ptr++ = 0.0;
        *ptr++ = (float) _activeVoxel[i];
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);

}

void VoxelObject::initShape(ObjectShape shape)
{
	srand(1973);
    switch (shape)
    {
        default:
        case VOXEL_CUBE:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize * _objectParams._cubeSize) + (y * _objectParams._cubeSize) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            _activeVoxel[i] = 1;
                            // Calculate center of voxels for use in VBO rendering
                            _pos[i*4] = _objectParams._origin.x + (x - _objectParams._cubeSize / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+1] = _objectParams._origin.y + (y - _objectParams._cubeSize / 2.0) *_objectParams. _voxelSize;
                            _pos[i*4+2] = _objectParams._origin.z + (z - _objectParams._cubeSize / 2.0) * _objectParams._voxelSize;
                            _pos[i*4+3] = 1.0f;
                        }
                    }
                }
            }
        }
        break;

        case VOXEL_PLANE:
        {
            // To fill in Later
        }
        break;
        case VOXEL_SPHERE:
        {
            for (unsigned int z = 0; z < _objectParams._cubeSize; z++)
            {
                for (unsigned int y = 0; y < _objectParams._cubeSize; y++)
                {
                    for (unsigned int x = 0; x < _objectParams._cubeSize; x++)
                    {
                        unsigned int i = (z*_objectParams._cubeSize * _objectParams._cubeSize) + (y * _objectParams._cubeSize) + x;

                        if (i < _objectParams._numVoxels)
                        {
                            float xPos = _objectParams._origin.x + (x - _objectParams._cubeSize / 2.0) * _objectParams._voxelSize;
                            float yPos = _objectParams._origin.y + (y - _objectParams._cubeSize / 2.0) *_objectParams. _voxelSize;
                            float zPos = _objectParams._origin.z + (z - _objectParams._cubeSize / 2.0) * _objectParams._voxelSize;
                            float radius = sqrt((xPos - _objectParams._origin.x) * (xPos - _objectParams._origin.x) + 
                                                (yPos - _objectParams._origin.y) * (yPos - _objectParams._origin.y) +
                                                (zPos - _objectParams._origin.z) * (zPos - _objectParams._origin.z));
                            if (radius <= (_objectParams._cubeSize * _objectParams._voxelSize) / 2.0) {
                                _activeVoxel[i] = 1;
                                // Calculate center of voxels for use in VBO rendering
                                _pos[i*4] = xPos;
                                _pos[i*4+1] = yPos;
                                _pos[i*4+2] = zPos;
                                _pos[i*4+3] = 1.0f;
                            } else {
                                _activeVoxel[i] = 0;
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    copyArrayToDevice((void *) _dev_activeVoxel, (void *) _activeVoxel, 0, _objectParams._numVoxels);

    // Copy position data to vbo
    unregisterGLBufferObject(_cuda_posvbo_resource);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _objectParams._numVoxels*4*sizeof(float), _pos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);

}

float* VoxelObject::getPosArray() {
    float *dPos = (float *) mapGLBufferObject(&_cuda_posvbo_resource);
    return dPos;
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