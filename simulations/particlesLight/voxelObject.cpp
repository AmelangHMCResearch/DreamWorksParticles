#include "voxelObject.h"
#include "particleSystem.cuh"

VoxelObject::VoxelObject(ObjectShape shape, float voxelSize, unsigned int cubeSize, float3 origin)
  :  _pos(0),
    _activeVoxel(0),
    _voxelSize(voxelSize),
	_cubeSize(cubeSize),  
    _numVoxels(cubeSize * cubeSize * cubeSize), 
    _origin(origin)
{
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
	unsigned int memSize = _numVoxels * sizeof(float);

    _pos = new float[4 * _numVoxels];
    _activeVoxel = new bool[_numVoxels];

    // Allocate active voxel array on GPU
    allocateArray((void **) &_dev_activeVoxel, memSize);

    // Create the VBO
    glGenBuffers(1, &_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferData(GL_ARRAY_BUFFER, _numVoxels * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);
 
    // Create the color buff
    _colorVBO = createVBO(_numVoxels * 4 * sizeof(float));
    registerGLBufferObject(_colorVBO, &_cuda_colorvbo_resource);
    // fill color buffer
    glBindBufferARB(GL_ARRAY_BUFFER, _colorVBO);
    float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    float *ptr = data;
    for (unsigned int i = 0; i < _numVoxels; i++)
    {
        *ptr++ = 1.0;
        *ptr++ = 0.0;
        *ptr++ = 0.0;
        *ptr++ = 1.0f;
    }
    glUnmapBufferARB(GL_ARRAY_BUFFER);

    // Place voxels at correct positions; 
    initShape(shape);
}

void VoxelObject::initShape(ObjectShape shape)
{
	srand(1973);
    switch (shape)
    {
        default:
        case VOXEL_CUBE:
        {
            for (unsigned int z = 0; z < _cubeSize; z++)
            {
                for (unsigned int y = 0; y < _cubeSize; y++)
                {
                    for (unsigned int x = 0; x < _cubeSize; x++)
                    {
                        unsigned int i = (z*_cubeSize * _cubeSize) + (y * _cubeSize) + x;

                        if (i < _numVoxels)
                        {
                            _activeVoxel[i] = 1;
                            // Calculate center of voxels for use in VBO rendering
                            _pos[i*4] = _origin.x + (x - _cubeSize / 2.0) * _voxelSize;
                            _pos[i*4+1] = _origin.y + (y - _cubeSize / 2.0) * _voxelSize;
                            _pos[i*4+2] = _origin.z + (z - _cubeSize / 2.0) * _voxelSize;
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
            // To Fill in Later
        }
        break;
    }

    // Copy position data to vbo
    unregisterGLBufferObject(_cuda_posvbo_resource);
    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _numVoxels*4*sizeof(float), _pos);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);

}

float* VoxelObject::getPosArray() {
    float *dPos = (float *) mapGLBufferObject(&_cuda_posvbo_resource);
    return dPos;
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