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

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool useOpenGL) :
    _systemInitialized(false),
    _usingOpenGL(useOpenGL),
    _shouldResort(true), 
    _dev_shouldResort(0),
    _numParticles(numParticles),
    _pos(0),
    _vel(0),
    _cellStart(0),
    _cellEnd(0),
    _numNeighbors(0),
    _dev_oldPos(0),
    _dev_vel(0),
    _dev_force(0),
    _dev_numNeighbors(0),
    _dev_cellStart(0),
    _dev_cellEnd(0),
    _gridSize(gridSize)
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
    _params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    float cellSize = 8.0f / (float) _gridSize.x;  // cell size equal to particle diameter
    _params.cellSize = make_float3(cellSize, cellSize, cellSize);

    _params.spring = 0.5f;
    _params.damping = 0.02f;
    _params.shear = 0.1f;
    _params.attraction = 0.0f;
    _params.boundaryDamping = -0.5f;
    _params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    _params.globalDamping = 1.0f;

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

    checkCudaErrors(cudaMalloc((void **)&_dev_shouldResort, sizeof(bool)));
    checkCudaErrors(cudaMemcpy(_dev_shouldResort, &_shouldResort, sizeof(bool), cudaMemcpyHostToDevice));

    if (_usingOpenGL)
    {
        _posVBO = createVBO(memSize);
        registerGLBufferObject(_posVBO, &_cuda_posvbo_resource);
    }

    allocateArray((void **)&_dev_oldPos, memSize);
    allocateArray((void **)&_dev_vel, memSize);
    allocateArray((void **)&_dev_force, memSize);
    checkCudaErrors(cudaMemset(_dev_force, 0, memSize));

    allocateArray((void **) &_dev_numNeighbors, (_numParticles+1)*sizeof(uint)); 
    checkCudaErrors(cudaMemset(_dev_numNeighbors, 0, (_numParticles + 1) * sizeof(uint)));

    allocateArray((void **)&_dev_cellIndex, _numParticles*sizeof(uint));
    allocateArray((void **)&_dev_particleIndex, _numParticles*sizeof(uint));

    allocateArray((void **)&_dev_cellStart, _numGridCells*sizeof(uint));
    allocateArray((void **)&_dev_cellEnd, _numGridCells*sizeof(uint));

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
    freeArray(_dev_oldPos);
    freeArray(_dev_force);
    freeArray(_dev_numNeighbors);

    freeArray(_dev_cellIndex);
    freeArray(_dev_particleIndex);
    freeArray(_dev_cellStart);
    freeArray(_dev_cellEnd);

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
                    _dev_oldPos,
                    _dev_shouldResort,
                    deltaTime,
                    _numParticles,
                    _timer);

    checkCudaErrors(cudaMemcpy((void *)&_shouldResort, _dev_shouldResort, sizeof(bool), cudaMemcpyDeviceToHost));

    if (_shouldResort) {

        // calculate grid hash
        calcCellIndices(_dev_cellIndex,
                        _dev_particleIndex,
                        dPos,
                        _numParticles,
                        _timer);
#if 1
    
        // sort particles based on hash
        sortParticles(_dev_cellIndex, 
                      _dev_particleIndex, 
                      _numParticles, 
                      _timer);

        float* tempPos;
        float* tempVel;
        allocateArray((void **)&tempPos, _numParticles*4*sizeof(float));
        allocateArray((void **)&tempVel, _numParticles*4*sizeof(float));
        /*checkCudaErrors(cudaMemcpy(tempPos, dPos, 4*sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(tempVel, _dev_vel, 4*sizeof(float), cudaMemcpyDeviceToDevice));*/

        copyArrays(dPos,
                   tempPos,
                   _dev_vel,
                   tempVel,
                   _numParticles,
                   _timer);
    
        // reorder particle arrays into sorted order and
        // find start and end of each cell
        reorderDataAndFindCellStart(_dev_cellStart,
                                    _dev_cellEnd,
                                    _dev_cellIndex,
                                    _dev_particleIndex,
                                    dPos,
                                    tempPos,
                                    _dev_oldPos,
                                    _dev_vel,
                                    tempVel,
                                    _numParticles,
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
    }

    // process collisions
    collide(dPos,
            _dev_vel,
            _dev_force,
            _dev_cellIndex,
            _dev_cellStart,
            _dev_cellEnd,
            _dev_numNeighbors,
            _numParticles,
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

void
ParticleSystem::reset(ParticleConfig config)
{
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