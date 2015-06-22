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

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"
#include "event_timer.h" 

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>
#include <vector>

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, float* rot, float* trans, bool useOpenGL) :
    _systemInitialized(false),
    _usingOpenGL(useOpenGL),
    _numParticles(numParticles),
    _numActiveParticles(0),
    _posAfterLastSortIsValid(false),
    _numTimesteps(0),
    _initialVel(0.3f),
    _pos(0),
    _vel(0),
    _cellStart(0),
    _cellEnd(0),
    _numNeighbors(0),
    _dev_posAfterLastSort(0),
    _dev_vel(0),
    _dev_force(0),
    _dev_numNeighbors(0),
    _dev_cellStart(0),
    _dev_cellEnd(0),
    _gridSize(gridSize),
    dummy_iterationsSinceLastResort(0)
{
    _timer = new EventTimer(5); 
    _numGridCells = _gridSize.x * _gridSize.y * _gridSize.z;

    // set simulation parameters
    _params.gridSize = _gridSize;
    _params.numCells = _numGridCells;
    _params.numBodies = _numParticles;

    _params.particleRadius = 1.0f / 64.0f;
    _params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
    _params.colliderRadius = 0.2f;

    _params.worldOrigin = make_float3(0.0f, 0.0f, 0.0f);
    float cellSize = 8.0f / (float) _gridSize.x;  // cell size equal to particle diameter
    _params.cellSize = make_float3(cellSize, cellSize, cellSize);

    _params.spring = 0.5f;
    _params.damping = 0.02f;
    _params.shear = 0.1f;
    _params.attraction = 0.0f;
    _params.boundaryDamping = -0.5f;
    _params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    _params.globalDamping = 1.0f;
    // fixed initial value for cell padding / movement threshold
    _params.movementThreshold = 0.2*_params.particleRadius;

    setRotation(rot);
    setTranslation(trans);

    _initialize();
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize()
{
    assert(!_systemInitialized);

    // allocate host storage
    _pos = new float[_numParticles*4];
    _vel = new float[_numParticles*4];
    memset(_pos, 0, _numParticles*4*sizeof(float));
    memset(_vel, 0, _numParticles*4*sizeof(float));

    _cellStart = new uint[_numGridCells];
    memset(_cellStart, 0, _numGridCells*sizeof(uint));

    _cellEnd = new uint[_numGridCells];
    memset(_cellEnd, 0, _numGridCells*sizeof(uint));

    // Allocate testing arrays here - to track num neighbors. 
    _numNeighbors = new uint[_numParticles + 1];
    memset(_numNeighbors, 0, (_numParticles+1)*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * _numParticles;

    if (_usingOpenGL)
    {
        _posVBO = createVBO(memSize);
        registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);
    }

    allocateArray((void **)&_dev_posAfterLastSort, memSize);
    allocateArray((void **)&_dev_vel, memSize);
    allocateArray((void **)&_dev_force, memSize);
    checkCudaErrors(cudaMemset(_dev_force, 0, memSize));

    allocateArray((void **) &_dev_numNeighbors, (_numParticles+1)*sizeof(uint)); 
    checkCudaErrors(cudaMemset(_dev_numNeighbors, 0, (_numParticles + 1) * sizeof(uint)));

    allocateArray((void **)&_dev_cellIndex, _numParticles*sizeof(uint));
    allocateArray((void **)&_dev_particleIndex, _numParticles*sizeof(uint));

    allocateArray((void **)&_dev_cellStart, _numGridCells*sizeof(uint));
    allocateArray((void **)&_dev_cellEnd, _numGridCells*sizeof(uint));

    allocateArray((void **)&_dev_pointHasMovedMoreThanThreshold, sizeof(bool));
    cudaMemset(_dev_pointHasMovedMoreThanThreshold, true, sizeof(bool));

    if (_usingOpenGL)
    {
        _colorVBO = createVBO(_numParticles*4*sizeof(float));
        registerGLBufferObject(_colorVBO, &_cuda_colorvbo_resource);

        // fill color buffer
        glBindBufferARB(GL_ARRAY_BUFFER, _colorVBO);
        float *data = (float *) glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i = 0; i < _numParticles; i++)
        {
            float t = i / (float) _numParticles;
#if 0
            *ptr++ = 1.0;
            *ptr++ = 0.0;
            *ptr++ = 0.0;
#else
            colorRamp(t, ptr);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBufferARB(GL_ARRAY_BUFFER);
    }

    setParameters(&_params);

    _systemInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(_systemInitialized);

    delete [] _vel;
    delete [] _pos;
    delete [] _cellStart;
    delete [] _cellEnd;

    delete [] _numNeighbors;
    delete _timer;

    freeArray(_dev_vel);
    freeArray(_dev_posAfterLastSort);
    freeArray(_dev_force);
    freeArray(_dev_numNeighbors);

    freeArray(_dev_cellIndex);
    freeArray(_dev_particleIndex);
    freeArray(_dev_cellStart);
    freeArray(_dev_cellEnd);
    freeArray(_dev_pointHasMovedMoreThanThreshold);

    if (_usingOpenGL)
    {
        unregisterGLBufferObject(_cuda_colorvbo_resource);
        unregisterGLBufferObject(_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&_posVBO);
        glDeleteBuffers(1, (const GLuint *)&_colorVBO);
    }
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(_systemInitialized);

    if (_numTimesteps * deltaTime * _initialVel > 2 * _params.particleRadius || _numActiveParticles == 0) {
        _numTimesteps = 0;
        float spoutRadius = 0.5f;
        float jitter = 0.1f;       // Room for jitter as a percentage of particle radius
        addParticles(spoutRadius, jitter);
    }
    ++_numTimesteps;

    float *dPos;

    if (_usingOpenGL)
    {
        dPos = (float *) mapGLBufferObject(&_cuda_posvbo_resource);
    }

    // update constants
    setParameters(&_params);

    // integrate
    integrateSystem(dPos,
                    _dev_vel,
                    _dev_force,
                    _dev_posAfterLastSort,
                    deltaTime,
                    _numActiveParticles,
                    _posAfterLastSortIsValid,
                    _dev_pointHasMovedMoreThanThreshold,
                    _timer);

    bool needToResort = checkForResort(_dev_pointHasMovedMoreThanThreshold);

    if (needToResort) {

        // calculate grid hash
        calcCellIndices(_dev_cellIndex,
                        _dev_particleIndex,
                        dPos,
                        _numActiveParticles,
                        _timer);
#if 1
    
        // sort particles based on hash
        sortParticles(_dev_cellIndex, 
                      _dev_particleIndex, 
                      _numActiveParticles, 
                      _timer);

        float* tempPos;
        float* tempVel;
        allocateArray((void **)&tempPos, _numActiveParticles*4*sizeof(float));
        allocateArray((void **)&tempVel, _numActiveParticles*4*sizeof(float));

        copyArrays(dPos,
                   tempPos,
                   _dev_vel,
                   tempVel,
                   _numActiveParticles,
                   _timer);
    
        // reorder particle arrays into sorted order and
        // find start and end of each cell
        reorderDataAndFindCellStart(_dev_cellStart,
                                    _dev_cellEnd,
                                    _dev_cellIndex,
                                    _dev_particleIndex,
                                    dPos,
                                    tempPos,
                                    _dev_posAfterLastSort,
                                    _dev_vel,
                                    tempVel,
                                    &_posAfterLastSortIsValid,
                                    _dev_pointHasMovedMoreThanThreshold,
                                    _numActiveParticles,
                                    _numGridCells,
                                    _timer);
        freeArray(tempPos);
        freeArray(tempVel);

#else
        // sort particles based on hash
        sortParticlesOnce(_dev_cellIndex, 
                          dPos,
                          _dev_vel,
                          _numParticles, 
                          _timer);
    
        // reorder particle arrays into sorted order and
        // find start and end of each cell
        findCellStart(_dev_cellStart,
                      _dev_cellEnd,
                      _dev_cellIndex,
                      dPos,
                      _dev_oldPos,
                      _numParticles,
                      _numGridCells,
                      _timer);

#endif
        // printf("Number of iterations since last sort = %d\n", dummy_iterationsSinceLastResort);
        dummy_iterationsSinceLastResort = 0;
    } else {
        ++dummy_iterationsSinceLastResort;
    }

    // process collisions
    collide(dPos,
            _dev_vel,
            _dev_force,
            _dev_cellIndex,
            _dev_cellStart,
            _dev_cellEnd,
            _dev_numNeighbors,
            _numActiveParticles,
            _numGridCells,
            _timer);
    

    /*checkCudaErrors(cudaMemcpy(_numNeighbors, _dev_numNeighbors, 
                               (_numParticles + 1)*sizeof(uint), cudaMemcpyDeviceToHost));*/

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (_usingOpenGL)
    {
        unmapGLBufferObject(_cuda_posvbo_resource);
    }
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(_cellStart, _dev_cellStart, 0, sizeof(uint)*_numGridCells);
    copyArrayFromDevice(_cellEnd, _dev_cellEnd, 0, sizeof(uint)*_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<_numGridCells; i++)
    {
        if (_cellStart[i] != 0xffffffff)
        {
            uint cellSize = _cellEnd[i] - _cellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(_pos, 0, &_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(_vel, _dev_vel, 0, sizeof(float)*4*count);

    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", _pos[i*4+0], _pos[i*4+1], _pos[i*4+2], _pos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", _vel[i*4+0], _vel[i*4+1], _vel[i*4+2], _vel[i*4+3]);
    }
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(_systemInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (_usingOpenGL)
                {
                    unregisterGLBufferObject(_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, _posVBO);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);

                    registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);

                    // force a resort because particles have moved
                    _posAfterLastSortIsValid = false;
                    cudaMemset(_dev_pointHasMovedMoreThanThreshold, true, sizeof(bool));
                }
            }
            break;

        case VELOCITY:
            copyArrayToDevice(_dev_vel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    _pos[i*4] = (spacing * x) + _params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    _pos[i*4+1] = (spacing * y) + _params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    _pos[i*4+2] = (spacing * z) + _params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    _pos[i*4+3] = 1.0f;
                    _vel[i*4] = 0.0f;
                    _vel[i*4+1] = 0.0f;
                    _vel[i*4+2] = 0.0f;
                    _vel[i*4+3] = 0.0f;
                }
            }
        }
    }
}

float findRadius(float2 radius) {
    return sqrt(radius.x * radius.x + radius.y * radius.y);
}

float3 rotatePoint(float3* rotMatrix, float3 pos) {
    float3 result;
    result.x = rotMatrix[0].x * pos.x + rotMatrix[0].y * pos.y + rotMatrix[0].z * pos.z;
    result.y = rotMatrix[1].x * pos.x + rotMatrix[1].y * pos.y + rotMatrix[1].z * pos.z;
    result.z = rotMatrix[2].x * pos.x + rotMatrix[2].y * pos.y + rotMatrix[2].z * pos.z;
    return result;
}

float3 crossProduct(float3 a, float3 b) {
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.z;
    return result;
}

float dotProduct(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

float3 scalarMult(float a, float3 b) {
    b.x = b.x * a;
    b.y = b.y * a;
    b.z = b.z * a;
    return b;
}

float3 operator+(const float3 & a, const float3 & b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

float3 operator-(const float3 & a, const float3 & b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

float3 operator*(const float alpha, const float3 & b) {
  return scalarMult(alpha, b);
}

float3 normalize(const float3 & a) {
  const float magnitude = sqrt(dotProduct(a, a));
  return scalarMult(1. / magnitude, a);
}

// Generates grid with side length radius based on hexagonal dense packing pattern
std::vector<float2> genParticlePos(float particleRadius, float radius)
{
    std::vector<float2> grid; 
    float cellWidth = 2 * particleRadius;
    float cellLength = 2 * sqrt(2 * particleRadius * particleRadius);
    uint gridWidth = ceil(2 * radius / cellWidth);
    uint gridLength = ceil(2 * radius / cellLength);
    for (int i = 0; i < gridLength; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            float2 newIntersection = make_float2(-1.0 * radius + j * cellWidth, -1.0 * radius + i * cellLength);
            grid.push_back(newIntersection);
            if (i > 0 && j > 0) {
                float2 center;
                center.x = newIntersection.x - 0.5 * cellWidth;
                center.y = newIntersection.y - 0.5 * cellLength;
                grid.push_back(center); 
            }
        }
    }
    std::vector<float2>::iterator i = grid.begin();
    while (i != grid.end()) {
        if (findRadius(*i) > radius) {
            i = grid.erase(i);
        } else {
            ++i;
        }
    }
    return grid;
}

void
ParticleSystem::addParticles(float spoutRadius, float jitter)
{

    const float3 unrotatedCameraPosition = -1. * _translation;
    float3 rotations = _rotation;
    rotations.x = rotations.x * (3.141592/180.0);
    rotations.y = rotations.y * (3.141592/180.0);

    // rotation matrices are the transpose (inverse) of what we apply to
    //  drawing.
    float3 xRotation[3] = {make_float3(1, 0, 0),
                          make_float3(0, cos(rotations.x), sin(rotations.x)),
                          make_float3(0, -1.0 * sin(rotations.x), cos(rotations.x))};
    float3 yRotation[3] = {make_float3(cos(rotations.y), 0, -1.0 * sin(rotations.y)),
                          make_float3(0, 1, 0),
                          make_float3(sin(rotations.y), 0, cos(rotations.y))};

    // we apply x, then y because of the rules of transpose.
    // when drawing, we do T*Rx*Ry.
    // we now want to do the inverse, which is the transpose of the rotations.
    // that means we need (Rx*Ry)^T, which is Ry^T*Rx^T.
    // so, we first apply transposed x rotation, then transposed y.
    const float3 cameraPosition =
      rotatePoint(yRotation, rotatePoint(xRotation, unrotatedCameraPosition));

    const std::vector<float2> positionsOfNewParticlesInThePlane =
      genParticlePos(_params.particleRadius * (1.0f + jitter), 0.15f);

    uint newNumParticles;
    uint amountToCopy;
    _spoutSize = positionsOfNewParticlesInThePlane.size();
    if (_numActiveParticles + _spoutSize < _numParticles) {
        newNumParticles = _numActiveParticles + _spoutSize;
        amountToCopy = _spoutSize;
    }
    else {
        newNumParticles = _numParticles;
        amountToCopy = _numParticles - _numActiveParticles;
    }

    const float3 cameraDirection = -1 * normalize(cameraPosition);
    // goal: make two axes within the plane of the spout's exit
    // strategy: use gram-schmidt orthogonalization
    float3 axis1 = make_float3(frand(), frand(), frand());
    // subtract off component in same direction as camera
    const float originalAxis1DotCameraDirection =
      dotProduct(axis1, cameraDirection);
    axis1 = axis1 - originalAxis1DotCameraDirection * cameraDirection;
    // axis1 is now orthogonal to cameraDirection
    axis1 = normalize(axis1);
    // axis1 is now normal and ready to use
    // now, form a second axis by taking the cross product of the two we have
    const float3 axis2 = crossProduct(cameraDirection, axis1);

    // use the camera direction as the initial velocity direction
    const float3 newVel = scalarMult(_initialVel, cameraDirection);
    for (int i = 0; i < newNumParticles - _numActiveParticles; ++i)
    {
        const float3 newParticlesPosition =
          cameraPosition +
          positionsOfNewParticlesInThePlane[i].x * axis1 + positionsOfNewParticlesInThePlane[i].y * axis1;
        _pos[i*4+0] = newParticlesPosition.x + frand() * jitter;
        _pos[i*4+1] = newParticlesPosition.y + frand() * jitter;
        _pos[i*4+2] = newParticlesPosition.z + frand() * jitter;
        _pos[i*4+3] = 1.0f;
        _vel[i*4+0] = 1.0 * newVel.x;
        _vel[i*4+1] = 1.0 * newVel.y;
        _vel[i*4+2] = 1.0 * newVel.z;
        _vel[i*4+3] = 0.0f;
    }

    setArray(POSITION, _pos, _numActiveParticles, amountToCopy);
    setArray(VELOCITY, _vel, _numActiveParticles, amountToCopy);
    _numActiveParticles = newNumParticles;
}

void
ParticleSystem::reset(ParticleConfig config)
{
    srand(1973);
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < _numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    _pos[p++] = 2 * (point[0] - 0.5f);
                    _pos[p++] = 2 * (point[1] - 0.5f);
                    _pos[p++] = 2 * (point[2] - 0.5f);
                    _pos[p++] = 1.0f; // radius
                    _vel[v++] = 0.0f;
                    _vel[v++] = 0.0f;
                    _vel[v++] = 0.0f;
                    _vel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = _params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) _numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, _params.particleRadius*2.0f, jitter, _numParticles);
            }
            break;
        case CONFIG_SPOUT:
            {
                // Write the particles to a location off screen, otherwise they default to
                // their old positions.
                int p = 0, v = 0;

                for (uint i=0; i < _numParticles; i++)
                {
                    _pos[4*i] = -15.0;
                    _pos[4*i + 1] = -15.0;
                    _pos[4*i + 2] = -15.0;
                    _pos[4*i + 3] = 1.0f; // radius
                    _vel[4*i] = 0.0f;
                    _vel[4*i + 1] = 0.0f;
                    _vel[4*i + 2] = 0.0f;
                    _vel[4*i + 3] = 0.0f;
                }
            }
            break;
    }
    setArray(POSITION, _pos, 0, _numParticles);
    setArray(VELOCITY, _vel, 0, _numParticles);

}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = _params.particleRadius*0.01f;

                if ((l <= _params.particleRadius*2.0f*r) && (index < _numParticles))
                {
                    _pos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    _pos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    _pos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    _pos[index*4+3] = pos[3];

                    _vel[index*4]   = vel[0];
                    _vel[index*4+1] = vel[1];
                    _vel[index*4+2] = vel[2];
                    _vel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, _pos, start, index);
    setArray(VELOCITY, _vel, start, index);
}
