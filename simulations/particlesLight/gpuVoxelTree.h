/* gpuVoxelTree.cpp
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

#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>
#include <array>

enum Status {
    ACTIVE,
    INACTIVE,
    DIG_DEEPER
};

// TODO: adapt to GPU ??
typedef std::array<float, 2> Point;

struct BoundingBox {
    Point lowerBoundary;
    Point upperBoundary;
};

// To hold the individual voxel data
template<typename DataType>
struct Voxel {
    float3 position;
    DataType  data;
};

// to allow for modular tree types
template<typename ChildNodeType, unsigned int numberOfChildrenPerSide>
class InternalNode
{
    public:
        InternalNode();
        ~InternalNode();

        Status getStatus();
    protected:
        static const unsigned int numberOfChildren = numberOfChildrenPerSide * numberOfChildrenPerSide;

        // to be changed to gpu data
        BoundingBox boundary;
        std::array<ChildNodeType, numberOfChildren> children;
        Status status;
};

// main tree class
template<typename ChildNodeType>
class RootNode
{
    public:
        RootNode();
        ~RootNode();

    protected:
        BoundingBox boundary;
        std::vector<ChildNodeType> children;
        Status status;

};



// helpful trees
template<typename T, unsigned int N1, unsigned int N2>
struct Tree3 {
    typedef RootNode< InternalNode< Voxel<float> , 4 > > Type;
};




