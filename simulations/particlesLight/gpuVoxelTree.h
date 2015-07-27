/* gpuVoxelTree.h
 *
 * Author: Zakkai Davidson - zdavidson@hmc.edu
 * Date: 30 Jun 2015
 * 
 * Purpose: The class included in this file serves to represent a voxel based
 *          3D model on the GPU similar to that of OpenVDB. The data structure
 *          consists of tiers, each of which splits up the active space of
 *          interest into different sized cells. The lowest level contains the
 *          actual voxel data. The design of this data structure is simply to
 *          answer the question "is particle x inside a voxel?", and thus it is
 *          meant to be used primarily with particle simulations on the GPU.
 *
 */

#ifndef GPUVOXELTREE_HAS_BEEN_INCLUDED
#define GPUVOXELTREE_HAS_BEEN_INCLUDED


// cuda
#include <cuda_runtime.h>
#include "vector_types.h"

// c++
#include <stdlib.h>
#include <vector>

enum Status {
    ACTIVE,
    INACTIVE,
    DIG_DEEPER
};

// TODO: adapt to GPU ??

struct BoundingBox {
    float3 lowerBoundary;
    float3 upperBoundary;
};

// To hold the individual voxel data
template<typename DataType>
struct Voxel {
    float3 position;
    DataType  data;
};

// to allow for modular tree types
template<typename ChildNodeType, unsigned int numberOfChildrenPerSide>
class InternalLevel
{
    public:
        InternalLevel();
        ~InternalLevel();

    protected:
        static const unsigned int numberOfChildren = numberOfChildrenPerSide * numberOfChildrenPerSide;

        // single values
        ChildNodeType* _dev_nextLevelStart; // where to find the next level of tree

        // arrays
        Status* _dev_childStatuses;
        unsigned int* _dev_childDelimeters;
};

// root of tree
template<typename ChildNodeType, unsigned int numberOfLevels>
class RootLevel
{
    public:
        RootLevel();
        ~RootLevel();

    protected:
        // single values
        BoundingBox* _dev_boundary; // boundary of complete geometry
        ChildNodeType* _dev_nextLevelStart; // where to find the next level of tree


        // arrays
        Status* _dev_childStatuses;
        unsigned int* _dev_childDelimeters;
};


// helpful trees
template<typename T, unsigned int N1, unsigned int N2>
struct Tree3 {
    typedef RootLevel< InternalLevel< Voxel<float> , 4 > , 3 > Type;
};

// TODO
// Now that we know the number of levels at the root node, the GPU kernel can iterate through all the
// levels by following the pointers. This way we only need to pass the top level object to the GPU (?)



// Note: I think that we should actually go with the class below instead of the templated version.
//       the only major difference from a usage perspective is a slightly less stdlib-y
//       constructor, but everything looks much better from both an implementation an performance
//       prospective. For example, we don't have to chase nearly as many pointers to get to the
//       voxel level as we would have while using the templated version.

class VoxelTree
{
    public: 
        VoxelTree(std::vector<unsigned int> numberOfCellsPerSideForEachLevel, float voxelSize);

        ~VoxelTree();

        void initializeTree(); // TODO: Needs arguments (input VDB?)
        unsigned int getNumberOfLevels();
        std::vector<unsigned int> getNumberOfCellsPerSideForLevel();
        void runCollisions(float *particlePos, 
                           float *particleVel, 
                           float  particleRadius,
                           unsigned int numParticles);

        // TODO: Remove
        static void test();

    private:
        // status checking functions
        bool isInitialized();
        bool hasVoxelData();
        

    protected:
        // CPU values
        bool _isInitialized;
        unsigned int _numberOfLevels;
        unsigned int _numVoxels; 
        BoundingBox _boundary; 
        std::vector<unsigned int> _numberOfCellsPerSideForLevel;
        float _voxelSize; 
};






#endif // GPUVOXELTREE_HAS_BEEN_INCLUDED


