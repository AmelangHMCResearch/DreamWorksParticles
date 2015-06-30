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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "voxelObject.h"
#include "vector_functions.h"
#include "event_timer.h" 

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint3 gridSize, bool useOpenGL, bool useObject);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };

        void update(float deltaTime, VoxelObject *voxelObject);
        void reset(ParticleConfig config);

        void   setArray(ParticleArray array, const float *data, int start, int count);

        int    getNumParticles() const
        {
            return _numParticles;
        }

        unsigned int getCurrentReadBuffer() const
        {
            return _posVBO;
        }
        unsigned int getColorBuffer()       const
        {
            return _colorVBO;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count);

        void setDamping(float x)
        {
            _params.globalDamping = x;
        }
        void setGravity(float x)
        {
            _params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setCollideSpring(float x)
        {
            _params.spring = x;
        }
        void setCollideDamping(float x)
        {
            _params.damping = x;
        }
        void setCollideShear(float x)
        {
            _params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            _params.attraction = x;
        }

        void setColliderPos(float3 x)
        {
            _params.colliderPos = x;
        }

        float getParticleRadius()
        {
            return _params.particleRadius;
        }
        float3 getColliderPos()
        {
            return _params.colliderPos;
        }
        float getColliderRadius()
        {
            return _params.colliderRadius;
        }
        uint3 getGridSize()
        {
            return _params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return _params.worldOrigin;
        }
        float3 getCellSize()
        {
            return _params.cellSize;
        }

        uint* getNumNeighbors()
        {
            return _numNeighbors;
        }

        void startTimer(uint timerNum)
        {
            _timer->startTimer(timerNum, false);
        }

        void stopTimer(uint timerNum)
        {
            _timer->stopTimer(timerNum, false);
        }

        float* getTime()
        {
            return _timer->getTimes();
        }

        void addSphere(int index, float *pos, float *vel, int r, float spacing);

    protected: // methods
        ParticleSystem() {}
        uint createVBO(uint size);

        void _initialize();
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);

    protected: // data
        bool _systemInitialized; 
        bool _usingOpenGL;
        bool _posAfterLastSortIsValid; 
        uint _numParticles;

        // CPU data - Do we even need this?
        float *_pos;              
        float *_vel;   
        uint  *_cellStart;
        uint  *_cellEnd;         

        uint *_numNeighbors;



        // GPU data
        float *_dev_posAfterLastSort;
        bool  *_dev_pointHasMovedMoreThanThreshold;
        float *_dev_vel;
        float *_dev_force;
        uint *_dev_numNeighbors;     // How many neighbors each particle has

        // grid data for sorting method
        uint *_dev_cellIndex;    // grid hash value for each particle
        uint *_dev_particleIndex;// particle index for each particle
        uint *_dev_cellStart;        // index of start of each cell in sorted list
        uint *_dev_cellEnd;          // index of end of cell

        uint   _posVBO;            // vertex buffer object for particle positions
        uint   _colorVBO;          // vertex buffer object for colors

        struct cudaGraphicsResource *_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
        struct cudaGraphicsResource *_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

        // params
        SimParams _params;
        uint3 _gridSize;
        uint _numGridCells;

        uint dummy_iterationsSinceLastResort;

        EventTimer* _timer;

};

#endif // __PARTICLESYSTEM_H__