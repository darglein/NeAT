/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the GPL v3 License.
* See LICENSE file for more information.
*/

#pragma once

#include "HyperTree.h"


struct TreeOptimizerParams
{
    int num_threads       = 16;
    bool use_saved_errors = true;
    int max_active_nodes  = 512;
    bool verbose          = false;

    double error_merge_factor = 1.1;
    double error_split_factor = 0.75;
};


namespace operations_research
{
class MPVariable;
class MPConstraint;
class MPSolver;
}  // namespace operations_research
// Optimizes the node structure of a given tree so that the reconstruction error is reduced
// but the number of nodes stays the same.
//
// This is achieved by mixed integer programming (MIP).
//
// An object of this class should not be reused!
// Recreate it every time you want to optimize the structure!
//
class HyperTreeStructureOptimizer
{
   public:
    // This structs contains the optimization variables and constraints for every active(!) node.
    struct PerNodeData
    {
        // Indicator if we want to split or stay the same or want to merge
        operations_research::MPVariable *I_split = nullptr, *I_none = nullptr, *I_grp = nullptr;

        // true if all siblings are active and have I_up set to 1
        // operations_research::MPConstraint* c_merge_split = nullptr;
    };

    std::vector<PerNodeData> data;


    HyperTreeStructureOptimizer(HyperTreeBase tree, TreeOptimizerParams params);

    // Returns true if a structure change has been applied
    bool OptimizeTree();

   private:
    void CreateVariables();
    void CreateConstraints();

    void CreateObjective();

    void ApplySplit();

    std::shared_ptr<operations_research::MPSolver> solver;
    HyperTreeBase tree;
    TreeOptimizerParams params;

    std::mt19937 gen = std::mt19937(67934867);
};