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

// c++
#include <stdlib.h>
#include <vector>
#include <array>

enum Status {
    ACTIVE,
    INACTIVE,
    DIG_DEEPER
};

// TODO: adapt to GPU ??
typedef std::array<float, 3> Point;

struct BoundingBox {
    Point lowerBoundary;
    Point upperBoundary;
};

// Note: I think that we should actually go with the class below instead of the templated version.
//       the only major difference from a usage perspective is a slightly less stdlib-y
//       constructor, but everything looks much better from both an implementation an performance
//       prospective. For example, we don't have to chase nearly as many pointers to get to the
//       voxel level as we would have while using the templated version.

class VoxelTree
{
    public: 
        VoxelTree(std::vector<unsigned int> numberOfCellsPerSideForEachLevel);

        ~VoxelTree();

        void initializeTree(); // TODO: Needs arguments (input VDB?)
        unsigned int getNumberOfLevels();
        std::vector<unsigned int> getNumberOfCellsPerSideForLevel();
        std::vector<std::vector<float> > getStatuses(); // Only to be used for debugging
        std::vector<std::vector<unsigned int> > getDelimiters(); // Only to be used for debugging
        void debugDisplay();

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
        BoundingBox _boundary;
        std::vector<unsigned int> _numberOfCellsPerSideForLevel;

        // scalar values
        unsigned int* _dev_numberOfLevels; // TODO: allocate in constant memory
        BoundingBox*  _dev_boundary; // TODO: allocate in constant memory

        // configuration data
        unsigned int*  _dev_numberOfCellsPerSideForLevel; // TODO: constant memory

        // data
        float** _dev_pointersToLevelStatuses; // TODO: store pointers to global (texture?) memory in constant memory
        unsigned int** _dev_pointersToLevelDelimiters; // TODO: store pointers to global (texture?) memory in constant memory
        float*  _dev_voxels; // TODO: global or texture memory      
};






#endif // GPUVOXELTREE_HAS_BEEN_INCLUDED


